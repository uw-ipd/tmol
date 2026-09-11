"""Measure retained annotation storage across six fitted Cartbonded databases.

Reported bytes cover arrays reachable from the caches, not process/GPU peaks.
Per-fit setup timings are diagnostic single samples, not a speedup benchmark.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import attr
import numpy as np
import torch

from tmol.database import ParameterDatabase, inject_residue_params
from tmol.io import extended_pose_stack_from_sequences
from tmol.score import AtomTypeDependentTerm
from tmol.score.cartbonded import CartBondedEnergyTerm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    revision = "88209b75f1d567b7473d71060d709bfcae13bc26"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:tmol/score/cartbonded/_cartbonded_energy_term.py"],
        text=True,
    )
    namespace = {
        "__name__": "tmol.score.cartbonded._cartbonded_energy_term",
        "__package__": "tmol.score.cartbonded",
    }
    exec(compile(source, "baseline_cart_cache.py", "exec"), namespace)
    db = ParameterDatabase.get_default()
    pose = extended_pose_stack_from_sequences(["AA"], device=device)
    parent = AtomTypeDependentTerm(db, device)
    parent.setup_packed_block_types(pose.packed_block_types)
    original = db.scoring.cartbonded.residue_params["ALA"]
    databases = [
        inject_residue_params(
            db,
            [],
            cartbonded_params={
                "ALA": attr.evolve(
                    original,
                    length_parameters=(
                        attr.evolve(original.length_parameters[0], x0=1.0 + 0.2 * i),
                        *original.length_parameters[1:],
                    ),
                )
            },
        )
        for i in range(6)
    ]

    def fresh():
        packed = copy.copy(pose.packed_block_types)
        packed.active_block_types = [copy.copy(bt) for bt in packed.active_block_types]
        for owner in (packed, *packed.active_block_types):
            for name in tuple(vars(owner)):
                if name.startswith("cartbonded_"):
                    delattr(owner, name)
        return packed

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    report, histories = {}, {}
    for label, cls in (
        ("before", namespace["CartBondedEnergyTerm"]),
        ("after", CartBondedEnergyTerm),
    ):
        packed = fresh()
        rows, history = [], []
        for database in databases:
            term = cls(database, device)
            synchronize()
            start = time.perf_counter()
            for bt in packed.active_block_types:
                term.setup_block_type(bt)
            term.setup_packed_block_types(packed)
            synchronize()
            elapsed = time.perf_counter() - start
            history.append(packed.cartbonded_annotations[term.hash])
            tensors = [packed.cartbonded_atom_is_rosetta]
            for annotation in packed.cartbonded_annotations.values():
                tensors.extend(
                    value
                    for value in attr.asdict(annotation, recurse=False).values()
                    if isinstance(value, torch.Tensor)
                )
            arrays = [
                value
                for bt in packed.active_block_types
                for annotation in bt.cartbonded_annotations.values()
                for value in attr.asdict(annotation, recurse=False).values()
                if isinstance(value, np.ndarray)
            ]
            rows.append(
                dict(
                    setup_seconds=elapsed,
                    packed_cache_entries=len(packed.cartbonded_annotations),
                    max_block_cache_entries=max(
                        len(bt.cartbonded_annotations)
                        for bt in packed.active_block_types
                    ),
                    packed_cache_tensor_bytes=sum(
                        {
                            t.data_ptr(): t.numel() * t.element_size() for t in tensors
                        }.values()
                    ),
                    block_cache_numpy_bytes=sum(
                        {
                            a.__array_interface__["data"][0]: a.nbytes for a in arrays
                        }.values()
                    ),
                )
            )
        report[label] = rows
        histories[label] = history
    for old, new in zip(histories["before"], histories["after"]):
        for field in attr.fields(type(old)):
            a, b = getattr(old, field.name), getattr(new, field.name)
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                assert a == b
    result = dict(
        baseline=revision,
        device=str(device),
        torch=torch.__version__,
        n_types=pose.packed_block_types.n_types,
        rows=report,
        all_annotation_tensors_exact_match=True,
        scope="cache-reachable array storage only; excludes Python dictionaries/objects, rendered modules, profiler-held snapshots, allocator/native process memory; timings are single-sample diagnostics",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
