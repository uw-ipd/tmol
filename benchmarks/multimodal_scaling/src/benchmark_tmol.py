from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from common import (
    SPEC,
    benchmark_callable,
    data_path,
    machine_metadata,
    process_peak_rss_bytes,
    read_manifest,
    sha256,
)


def imports():
    import torch
    import tmol
    from tmol.database import ParameterDatabase
    from tmol.io import pose_stack_from_biotite, pose_stack_from_pdb
    from tmol.score import beta2016_score_function

    if not callable(pose_stack_from_biotite):
        pose_stack_from_biotite = pose_stack_from_biotite.pose_stack_from_biotite

    try:
        from tmol.pose import PoseStackBuilder
    except ImportError:
        from tmol.pose.pose_stack_builder import PoseStackBuilder

    return (
        torch,
        tmol,
        ParameterDatabase,
        pose_stack_from_biotite,
        pose_stack_from_pdb,
        beta2016_score_function,
        PoseStackBuilder,
    )


def select(dataset_id: str, modality: str | None = None) -> dict[str, str]:
    matches = [
        row
        for row in read_manifest()
        if row["dataset_id"] == dataset_id
        and row["status"] == "ok"
        and (modality is None or row["modality"] == modality)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one included dataset named {dataset_id!r}, got {len(matches)}"
        )
    return matches[0]


def load_pose(row: dict[str, str], device):
    (
        torch,
        _,
        ParameterDatabase,
        pose_stack_from_biotite,
        pose_stack_from_pdb,
        _,
        _,
    ) = imports()
    path = data_path(row["structure_path"])
    if row["modality"] != "protein_ligand":
        return pose_stack_from_pdb(
            path.read_text(), device
        ), ParameterDatabase.get_default()

    import biotite.structure as struc
    import biotite.structure.io

    structure = biotite.structure.io.load_structure(
        str(path), model=1, include_bonds=True
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    pose, context = pose_stack_from_biotite(
        structure,
        device,
        prepare_ligands=True,
        ligand_params_files=[str(data_path(row["ligand_tmol_params"]))],
        no_optH=True,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    return pose, context.parameter_database


def replicated_pose(row: dict[str, str], device, batch_size: int):
    *_, PoseStackBuilder = imports()
    pose, database = load_pose(row, device)
    loaded_blocks = int((pose.block_type_ind64[0] >= 0).sum())
    if loaded_blocks != int(row["residues"]):
        raise RuntimeError(
            f"loaded {loaded_blocks} blocks; manifest records {row['residues']} residues"
        )
    if batch_size == 1:
        return pose, database
    return PoseStackBuilder.from_poses([pose] * batch_size, device), database


def score_terms(sfxn, scorer, coords) -> dict[str, float]:
    unweighted = scorer.unweighted_scores(coords).detach()
    weights = sfxn.weights_tensor().detach()
    return {
        score_type.name: float(weights[index]) * float(unweighted[index, 0])
        for index, score_type in enumerate(sfxn.all_score_types())
    }


def benchmark_score(
    row,
    device_name: str,
    batch_size: int,
    gradient: bool,
    cuda_execution: str,
):
    torch, tmol, *_, beta2016_score_function, _ = imports()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_name == "cuda"
        else torch.device("cpu")
    )
    if device_name == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if device_name == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    setup_start = time.perf_counter()
    pose, database = replicated_pose(row, device, batch_size)
    sfxn = beta2016_score_function(device, param_db=database)
    cuda_graph_mode = None
    if device_name == "cuda" and cuda_execution != "eager":
        cuda_graph_mode = "forward_backward" if gradient else "forward"
    try:
        scorer = sfxn.render_whole_pose_scoring_module(
            pose, cuda_graph=cuda_graph_mode or False
        )
    except TypeError:
        # Historical releases predate reusable whole-pose CUDA graphs.
        cuda_graph_mode = None
        scorer = sfxn.render_whole_pose_scoring_module(pose)
    coords = pose.coords.detach().clone()
    if gradient:
        coords.requires_grad_(True)

        def operation():
            score = scorer(coords).sum()
            torch.autograd.grad(score, coords)

    else:

        def operation():
            scorer(coords)

    if device_name == "cuda":
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    timings, iterations = benchmark_callable(operation, device=device_name)
    validation_coords = pose.coords.detach().clone().requires_grad_(gradient)
    validation_score = scorer(validation_coords)
    if not torch.isfinite(validation_score).all():
        raise RuntimeError("non-finite score")
    validation_gradient_norm = None
    if gradient:
        (validation_gradient,) = torch.autograd.grad(
            validation_score.sum(), validation_coords
        )
        if not torch.isfinite(validation_gradient).all():
            raise RuntimeError("non-finite coordinate gradient")
        validation_gradient_norm = float(validation_gradient.norm())
    terms = score_terms(sfxn, scorer, coords.detach()) if batch_size == 1 else {}
    gpu_properties = None
    if device_name == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_properties = {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "seconds_per_batch_samples": timings,
        "seconds_per_structure_samples": [value / batch_size for value in timings],
        "iterations_per_sample": iterations,
        "setup_seconds": setup_seconds,
        "score_terms": terms,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device_name == "cuda"
        else None,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "tmol_import_version": getattr(tmol, "__version__", None),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_graph_mode": cuda_graph_mode,
        "torch_num_threads": torch.get_num_threads(),
        "gpu_properties": gpu_properties,
        "validation_score_mean": float(validation_score.detach().mean()),
        "validation_gradient_norm": validation_gradient_norm,
    }


def benchmark_fastrelax(row, device_name: str, batch_size: int):
    torch, tmol, *_, beta2016_score_function, _ = imports()
    try:
        from tmol import fast_relax
        from tmol.kinematics import CartesianMoveMap, FoldForest
        from tmol.pack import PackerPalette
    except ImportError:
        from tmol.relax.fast_relax import fast_relax
        from tmol.kinematics.move_map import CartesianMoveMap
        from tmol.kinematics.fold_forest import FoldForest
        from tmol.pack.packer_task import PackerPalette

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_name == "cuda"
        else torch.device("cpu")
    )
    if device_name == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if device_name == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    setup_start = time.perf_counter()
    pose, database = replicated_pose(row, device, batch_size)
    sfxn = beta2016_score_function(device, param_db=database)
    move_map = CartesianMoveMap()
    fold_forest = FoldForest.reasonable_fold_forest(pose)
    palette = PackerPalette()
    if device_name == "cuda":
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    initial_score = float(
        sfxn.render_whole_pose_scoring_module(pose)(pose.coords).detach().mean()
    )

    # FastRelax is a stochastic workflow. Time one reproducibly seeded run per
    # matrix point; setup and pose construction remain outside the interval.
    torch.manual_seed(SPEC["seed"])
    start = time.perf_counter()
    relaxed = fast_relax(
        pose,
        sfxn,
        palette,
        move_map,
        fold_forest,
        num_repeats=SPEC["fastrelax_repeats"],
        verbose=False,
    )
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if relaxed is None:
        raise RuntimeError("fast_relax returned None")
    final_score = float(
        sfxn.render_whole_pose_scoring_module(relaxed)(relaxed.coords).detach().mean()
    )
    return {
        "seconds_per_batch_samples": [elapsed],
        "seconds_per_structure_samples": [elapsed / batch_size],
        "iterations_per_sample": 1,
        "setup_seconds": setup_seconds,
        "score_terms": {},
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if device_name == "cuda"
        else None,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "tmol_import_version": getattr(tmol, "__version__", None),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "gpu_properties": (
            {
                "name": torch.cuda.get_device_properties(device).name,
                "total_memory": torch.cuda.get_device_properties(device).total_memory,
                "compute_capability": (
                    f"{torch.cuda.get_device_properties(device).major}."
                    f"{torch.cuda.get_device_properties(device).minor}"
                ),
            }
            if device_name == "cuda"
            else None
        ),
        "validation_score_mean": final_score,
        "initial_score_mean": initial_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--modality", choices=("protein", "protein_ligand", "protein_nucleic")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--protocol", choices=("score", "score_gradient", "fastrelax"), required=True
    )
    parser.add_argument("--tmol-version", required=True)
    parser.add_argument("--tmol-commit", required=True)
    parser.add_argument(
        "--cuda-execution",
        choices=("auto", "eager", "graph"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = select(args.dataset, args.modality)
    result = {
        "engine": "tmol",
        "engine_version": args.tmol_version,
        "engine_commit": args.tmol_commit,
        "protocol": args.protocol,
        "device": args.device,
        "batch_size": args.batch_size,
        "modality": row["modality"],
        "dataset_id": row["dataset_id"],
        "residues": int(row["residues"]),
        "polymer_residues": sum(
            int(row[key])
            for key in ("protein_residues", "dna_residues", "rna_residues")
        ),
        "atoms": int(row["atoms"]),
        "status": "ok",
        "error": "",
        "requested_cuda_execution": args.cuda_execution,
        "score_function": "beta2016",
        "atom_type_set": "tmol_default",
        "parameter_file_sha256": (
            sha256(data_path(row["ligand_tmol_params"]))
            if row["modality"] == "protein_ligand"
            else None
        ),
        "machine": machine_metadata(),
    }
    try:
        if args.protocol == "fastrelax":
            result.update(benchmark_fastrelax(row, args.device, args.batch_size))
        else:
            result.update(
                benchmark_score(
                    row,
                    args.device,
                    args.batch_size,
                    gradient=args.protocol == "score_gradient",
                    cuda_execution=args.cuda_execution,
                )
            )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        try:
            import torch

            if args.device == "cuda":
                result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        except Exception:
            pass
        result["process_peak_rss_bytes"] = process_peak_rss_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
