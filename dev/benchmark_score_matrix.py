#!/usr/bin/env python3
"""Benchmark tmol scoring on representative biomolecular systems.

This runner keeps structure preparation, score-function rendering, and steady-
state scoring as separate measurements.  It is intentionally independent of
pytest-benchmark so the same command can run under Nsight Systems/Compute.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite, pose_stack_from_pdb
from tmol.pose import PoseStack, PoseStackBuilder
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path


@dataclass(frozen=True)
class SystemSpec:
    kind: str
    path: str
    ligand_params: str | None = None


@dataclass(frozen=True)
class Timing:
    system: str
    kind: str
    device: str
    cuda_graph: str
    batch: int
    residues: int
    padded_atoms: int
    prepare_s: float
    batch_s: float
    render_s: float
    graph_setup_s: float
    forward_ms: float
    forward_backward_ms: float
    peak_memory_mb: float | None


SYSTEMS = {
    "protein-76": SystemSpec("protein", "pdb/1ubq.pdb"),
    "protein-300": SystemSpec("protein", "pdb/bysize_300_res_6f8b.pdb"),
    "protein-600": SystemSpec("protein", "pdb/bysize_600_res_5m4a.pdb"),
    "dna-24": SystemSpec("dna", "pdb/1bna.pdb"),
    "rna-42": SystemSpec("rna", "pdb/3zp8.pdb"),
    "protein-dna": SystemSpec("protein-dna", "pdb/1ysa.pdb"),
    "ligand-hsp90": SystemSpec(
        "protein-ligand",
        "protein_ligand_test/hsp90.tmol.nomin.cif",
        "protein_ligand_test/hsp90.xtal-lig.mmff94.tmol",
    ),
    "ligand-hivrt": SystemSpec(
        "protein-ligand",
        "protein_ligand_test/hivrt.tmol.nomin.cif",
        "protein_ligand_test/hivrt.xtal-lig.mmff94.tmol",
    ),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(device: torch.device, operation: Callable[[], object]) -> float:
    synchronize(device)
    start = time.perf_counter()
    operation()
    synchronize(device)
    return time.perf_counter() - start


def load_system(
    spec: SystemSpec, device: torch.device, optimize_hydrogens: bool
) -> tuple[PoseStack, ParameterDatabase]:
    path = data_path(spec.path)
    if spec.ligand_params is None:
        return (
            pose_stack_from_pdb(path.read_text(), device),
            ParameterDatabase.get_default(),
        )

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
        ligand_params_files=[str(data_path(spec.ligand_params))],
        no_optH=not optimize_hydrogens,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    return pose, context.parameter_database


def median_seconds(
    operation: Callable[[], object], device: torch.device, warmup: int, rounds: int
) -> float:
    for _ in range(warmup):
        operation()
    synchronize(device)
    return statistics.median(elapsed(device, operation) for _ in range(rounds))


def benchmark_system(
    name: str,
    spec: SystemSpec,
    device: torch.device,
    batch: int,
    optimize_hydrogens: bool,
    warmup: int,
    rounds: int,
    cuda_graph: str,
) -> Timing:
    holder: dict[str, object] = {}

    def prepare() -> None:
        holder["pose"], holder["database"] = load_system(
            spec, device, optimize_hydrogens
        )

    prepare_s = elapsed(device, prepare)
    pose = holder["pose"]
    database = holder["database"]
    assert isinstance(pose, PoseStack)
    assert isinstance(database, ParameterDatabase)

    batched_holder: dict[str, PoseStack] = {}

    def make_batch() -> None:
        batched_holder["pose"] = PoseStackBuilder.from_poses([pose] * batch, device)

    batch_s = elapsed(device, make_batch)
    batched = batched_holder["pose"]
    sfxn = beta2016_score_function(device, param_db=database)
    scorer_holder: dict[str, object] = {}
    render_s = elapsed(
        device,
        lambda: scorer_holder.update(
            scorer=sfxn.render_whole_pose_scoring_module(batched)
        ),
    )
    scorer = scorer_holder["scorer"]
    graph_setup_s = 0.0
    if cuda_graph != "none":
        graph_setup_s = elapsed(
            device,
            lambda: scorer.enable_cuda_graphs(batched.coords, mode=cuda_graph),
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    coords = batched.coords.detach()

    def forward() -> None:
        with torch.no_grad():
            scorer(coords).sum()

    forward_s = median_seconds(forward, device, warmup, rounds)

    def forward_backward() -> None:
        grad_coords = coords.detach().requires_grad_(True)
        scorer(grad_coords).sum().backward()

    full_s = median_seconds(forward_backward, device, warmup, rounds)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else None
    )
    residues = int((batched.block_type_ind[0] >= 0).sum())

    return Timing(
        system=name,
        kind=spec.kind,
        device=str(device),
        cuda_graph=cuda_graph,
        batch=batch,
        residues=residues,
        padded_atoms=batched.coords.shape[1],
        prepare_s=prepare_s,
        batch_s=batch_s,
        render_s=render_s,
        graph_setup_s=graph_setup_s,
        forward_ms=forward_s * 1000,
        forward_backward_ms=full_s * 1000,
        peak_memory_mb=peak_memory_mb,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--systems", nargs="+", choices=sorted(SYSTEMS), default=list(SYSTEMS)
    )
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 8))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--cuda-graph",
        choices=("none", "forward", "forward_backward", "both"),
        default="none",
    )
    parser.add_argument(
        "--no-opt-h",
        action="store_true",
        help="skip hydrogen optimization during ligand pose preparation",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1 or args.warmup < 0 or args.rounds < 1:
        raise SystemExit("cpu-threads and rounds must be positive; warmup may be zero")
    if any(batch < 1 for batch in args.batches):
        raise SystemExit("batch sizes must be positive")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device.type != "cuda" and args.cuda_graph != "none":
        raise SystemExit("--cuda-graph requires --device cuda")

    rows = []
    for name in args.systems:
        for batch in args.batches:
            row = benchmark_system(
                name,
                SYSTEMS[name],
                device,
                batch,
                not args.no_opt_h,
                args.warmup,
                args.rounds,
                args.cuda_graph,
            )
            rows.append(asdict(row))
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
