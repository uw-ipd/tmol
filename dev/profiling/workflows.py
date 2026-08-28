#!/usr/bin/env python3
"""Benchmark representative end-to-end TMol GPU workflows."""

# ruff: noqa: E402 -- import the checkout containing this developer script
# flake8: noqa

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import tmol
from tmol.database import ParameterDatabase
from tmol.io import (
    build_context_from_biotite,
    pose_stack_from_biotite,
    pose_stack_from_pdb,
)
from tmol.kinematics import CartesianMoveMap, FoldForest, MoveMap
from tmol.optimization import run_cart_min, run_kin_min
from tmol.pack import PackerPalette
from tmol.pose import PoseStackBuilder
from tmol.relax import fast_relax
from tmol.score import beta2016_score_function
from tmol.utility._nvtx import nvtx_range

DATA = ROOT / "tmol" / "tests" / "data"
SEED = 20260827


@dataclass(frozen=True)
class SystemSpec:
    path: str
    kind: str = "pdb"
    ligand_params: str | None = None


SYSTEMS = {
    "protein015": SystemSpec("pdb/bysize_015_res_1lu6.pdb"),
    "protein100": SystemSpec("pdb/bysize_100_res_5umr.pdb"),
    "protein400": SystemSpec("pdb/bysize_400_res_6azu.pdb"),
    "dna24": SystemSpec("cif/1BNA.cif", "polymer_cif"),
    "rna33": SystemSpec("cif/1EHT.cif", "polymer_cif"),
    "protein_dna": SystemSpec("cif/1HDD.cif", "polymer_cif"),
    "protein_ligand": SystemSpec(
        "protein_ligand_test/ada.tmol.nomin.cif",
        "ligand_cif",
        "protein_ligand_test/ada.xtal-lig.mmff94.tmol",
    ),
}

WORKFLOWS = (
    "score",
    "score_grad",
    "score_graph",
    "score_graph_grad",
    "cart_min",
    "kin_min",
    "fast_relax",
)


def _load_system(name: str, batch: int, device: torch.device):
    spec = SYSTEMS[name]
    path = DATA / spec.path
    param_db = None

    if spec.kind == "pdb":
        pose = pose_stack_from_pdb(str(path), device)
    else:
        import biotite.structure as struc
        from biotite.structure.io import load_structure

        structure = load_structure(str(path), model=1, include_bonds=True)
        if isinstance(structure, struc.AtomArrayStack):
            structure = structure[0]
        if spec.kind == "polymer_cif":
            structure = structure[~structure.hetero]
            pose = pose_stack_from_biotite(structure, device, no_optH=True)
        else:
            context = build_context_from_biotite(
                structure,
                device,
                prepare_ligands=True,
                ligand_params_files=[str(DATA / spec.ligand_params)],
                param_db=ParameterDatabase.get_default(),
                sample_proton_chi=False,
            )
            pose = pose_stack_from_biotite(
                structure, device, context=context, no_optH=True
            )
            param_db = context.parameter_database

    if batch > 1:
        pose = PoseStackBuilder.from_poses([pose] * batch, device)
    return pose, beta2016_score_function(device, param_db=param_db)


def _score_workload(pose, score_function, with_grad: bool, cuda_graph: bool = False):
    graph_mode = "forward_backward" if with_grad else "forward"
    scorer = score_function.render_whole_pose_scoring_module(
        pose, cuda_graph=graph_mode if cuda_graph else False
    )
    coords = pose.coords.detach().clone().requires_grad_(with_grad)

    if with_grad:

        def run():
            coords.grad = None
            scorer(coords).sum().backward()

    else:

        def run():
            with torch.no_grad():
                scorer(coords)

    return run


def _cart_min_workload(pose, score_function, max_iter: int):
    def run():
        result = run_cart_min(
            pose.clone(), score_function, optimizer_kwargs={"max_iter": max_iter}
        )
        if not torch.isfinite(result.coords[result.real_atoms]).all():
            raise RuntimeError("Cartesian minimization produced non-finite coordinates")

    return run


def _kin_min_workload(pose, score_function, max_iter: int):
    fold_forest = FoldForest.reasonable_fold_forest(pose)
    move_map = MoveMap.from_pose_stack(pose)
    move_map.move_all_named_torsions = True
    move_map.move_all_jumps = False
    move_map.move_all_root_jumps = False

    def run():
        result = run_kin_min(
            pose.clone(),
            score_function,
            fold_forest,
            move_map,
            optimizer_kwargs={"max_iter": max_iter},
        )
        if not torch.isfinite(result.coords[result.real_atoms]).all():
            raise RuntimeError("Kinematic minimization produced non-finite coordinates")

    return run


def _fast_relax_workload(pose, score_function, max_iter: int):
    fold_forest = FoldForest.reasonable_fold_forest(pose)

    def minimize(stage_pose, stage_score_function, **_):
        return run_cart_min(
            stage_pose,
            stage_score_function,
            optimizer_kwargs={"max_iter": max_iter},
        )

    def run():
        result = fast_relax(
            pose.clone(),
            score_function,
            PackerPalette(),
            CartesianMoveMap(),
            fold_forest,
            num_repeats=1,
            min_fn=minimize,
        )
        if not torch.isfinite(result.coords[result.real_atoms]).all():
            raise RuntimeError("FastRelax produced non-finite coordinates")

    return run


def _workload(pose, score_function, workflow: str, max_iter: int):
    if workflow == "score":
        return _score_workload(pose, score_function, False)
    if workflow == "score_grad":
        return _score_workload(pose, score_function, True)
    if workflow == "score_graph":
        return _score_workload(pose, score_function, False, True)
    if workflow == "score_graph_grad":
        return _score_workload(pose, score_function, True, True)
    if workflow == "cart_min":
        return _cart_min_workload(pose, score_function, max_iter)
    if workflow == "kin_min":
        return _kin_min_workload(pose, score_function, max_iter)
    if workflow == "fast_relax":
        return _fast_relax_workload(pose, score_function, max_iter)
    raise ValueError(workflow)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def _nvidia_driver() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("workflow profiling requires a CUDA GPU")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    case = f"{args.workflow}-{args.system}-b{args.batch}"

    with nvtx_range(f"{case}/setup"):
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        pose, score_function = _load_system(args.system, args.batch, device)
        torch.cuda.synchronize()
        system_setup_ms = (time.perf_counter() - setup_start) * 1e3

        workload_start = time.perf_counter()
        workload = _workload(pose, score_function, args.workflow, args.max_iter)
        torch.cuda.synchronize()
        workload_setup_ms = (time.perf_counter() - workload_start) * 1e3

    with nvtx_range(f"{case}/warmup"):
        for _ in range(args.warmup):
            workload()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    memory_before = torch.cuda.memory_allocated(device)
    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()

    durations = []
    try:
        with nvtx_range(f"{case}/measure"):
            for iteration in range(args.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with nvtx_range(f"{case}/iteration-{iteration}"):
                    workload()
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1e3)
    finally:
        if args.capture:
            torch.cuda.cudart().cudaProfilerStop()

    result = {
        "case": case,
        "workflow": args.workflow,
        "system": args.system,
        "batch": args.batch,
        "max_iter": args.max_iter,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "system_setup_ms": system_setup_ms,
        "workload_setup_ms": workload_setup_ms,
        "duration_ms": durations,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
        "min_ms": min(durations),
        "p95_ms": _percentile(durations, 95),
        "memory_before_bytes": memory_before,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_delta_bytes": torch.cuda.max_memory_allocated(device)
        - memory_before,
        "poses": pose.n_poses,
        "blocks_per_pose": int((pose.block_type_ind64[0] >= 0).sum()),
        "atoms_per_pose": int(pose.real_atoms[0].sum()),
        "environment": {
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tmol": tmol.__version__,
            "tmol_path": tmol.__file__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": ".".join(
                map(str, torch.cuda.get_device_capability(device))
            ),
            "driver": _nvidia_driver(),
            "hostname": platform.node(),
            "tmol_use_jit": os.environ.get("TMOL_USE_JIT"),
            "apptainer_container": os.environ.get("APPTAINER_CONTAINER"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "system_spec": asdict(SYSTEMS[args.system]),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=WORKFLOWS, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.max_iter < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error(
            "batch, max-iter, and iterations must be positive; warmup non-negative"
        )
    if args.workflow == "fast_relax" and args.system not in {
        "protein015",
        "protein100",
        "protein400",
    }:
        parser.error("the FastRelax matrix currently covers canonical proteins only")
    return args


def main():
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
