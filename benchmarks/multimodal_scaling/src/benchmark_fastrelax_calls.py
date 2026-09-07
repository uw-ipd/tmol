"""Measure FastRelax while counting whole-pose scorer invocations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import benchmark_tmol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import tmol.pack._pack_rotamers as pack_rotamers_module
    import tmol.relax._fast_relax as fast_relax_module
    from tmol.score._score_function import WholePoseScoringModule

    original_call = WholePoseScoringModule.__call__
    original_pack = fast_relax_module.pack_rotamers
    original_minimize = fast_relax_module._DefaultCartesianMinimizer.__call__
    original_build_rotamers = pack_rotamers_module.build_rotamers
    original_calculate_energies = pack_rotamers_module._calculate_packer_energies
    original_anneal = pack_rotamers_module.run_simulated_annealing
    original_assign = pack_rotamers_module.impose_top_rotamer_assignments
    counts = {
        "total": 0,
        "grad_enabled": 0,
        "coords_require_grad": 0,
        "packing_calls": 0,
        "packing_seconds": 0.0,
        "minimization_calls": 0,
        "minimization_seconds": 0.0,
        "rotamer_build_calls": 0,
        "rotamer_build_seconds": 0.0,
        "packer_energy_calls": 0,
        "packer_energy_seconds": 0.0,
        "annealing_calls": 0,
        "annealing_seconds": 0.0,
        "assignment_calls": 0,
        "assignment_seconds": 0.0,
    }

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def counted_call(self, coords, *call_args, **call_kwargs):
        counts["total"] += 1
        counts["grad_enabled"] += int(torch.is_grad_enabled())
        counts["coords_require_grad"] += int(coords.requires_grad)
        return original_call(self, coords, *call_args, **call_kwargs)

    def counted_pack(*call_args, **call_kwargs):
        synchronize()
        start = time.perf_counter()
        result = original_pack(*call_args, **call_kwargs)
        synchronize()
        counts["packing_calls"] += 1
        counts["packing_seconds"] += time.perf_counter() - start
        return result

    def counted_minimize(self, *call_args, **call_kwargs):
        synchronize()
        start = time.perf_counter()
        result = original_minimize(self, *call_args, **call_kwargs)
        synchronize()
        counts["minimization_calls"] += 1
        counts["minimization_seconds"] += time.perf_counter() - start
        return result

    def timed_subphase(original, calls_key, seconds_key):
        def wrapped(*call_args, **call_kwargs):
            synchronize()
            start = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            synchronize()
            counts[calls_key] += 1
            counts[seconds_key] += time.perf_counter() - start
            return result

        return wrapped

    WholePoseScoringModule.__call__ = counted_call
    fast_relax_module.pack_rotamers = counted_pack
    fast_relax_module._DefaultCartesianMinimizer.__call__ = counted_minimize
    pack_rotamers_module.build_rotamers = timed_subphase(
        original_build_rotamers, "rotamer_build_calls", "rotamer_build_seconds"
    )
    pack_rotamers_module._calculate_packer_energies = timed_subphase(
        original_calculate_energies, "packer_energy_calls", "packer_energy_seconds"
    )
    pack_rotamers_module.run_simulated_annealing = timed_subphase(
        original_anneal, "annealing_calls", "annealing_seconds"
    )
    pack_rotamers_module.impose_top_rotamer_assignments = timed_subphase(
        original_assign, "assignment_calls", "assignment_seconds"
    )
    try:
        row = benchmark_tmol.select(args.dataset, args.modality)
        result = benchmark_tmol.benchmark_fastrelax(row, args.device, args.batch_size)
    finally:
        WholePoseScoringModule.__call__ = original_call
        fast_relax_module.pack_rotamers = original_pack
        fast_relax_module._DefaultCartesianMinimizer.__call__ = original_minimize
        pack_rotamers_module.build_rotamers = original_build_rotamers
        pack_rotamers_module._calculate_packer_energies = original_calculate_energies
        pack_rotamers_module.run_simulated_annealing = original_anneal
        pack_rotamers_module.impose_top_rotamer_assignments = original_assign

    result.update(
        {
            "engine": "tmol",
            "engine_version": args.label,
            "engine_commit": args.commit,
            "protocol": "fastrelax",
            "device": args.device,
            "batch_size": args.batch_size,
            "modality": args.modality,
            "dataset_id": args.dataset,
            "polymer_residues": sum(
                int(row[key])
                for key in ("protein_residues", "dna_residues", "rna_residues")
            ),
            "whole_pose_call_counts": counts,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
