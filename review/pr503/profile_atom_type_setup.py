"""Paired shared atom-type setup, including native scalar-read counts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.database import ParameterDatabase
from tmol.io import extended_pose_stack_from_sequences
from tmol.score import AtomTypeDependentTerm

FIELDS = (
    "atom_types",
    "n_heavy_atoms",
    "heavy_atom_inds",
    "atom_unique_ids",
    "atom_wildcard_ids",
    "atom_cross_ids",
    "atom_unique_id_index",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    revision = "88209b75f1d567b7473d71060d709bfcae13bc26"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:tmol/score/_atom_type_dependent_term.py"],
        text=True,
    )
    namespace = {
        "__name__": "tmol.score._atom_type_dependent_term",
        "__package__": "tmol.score",
    }
    exec(compile(source, "baseline_atom_type_setup.py", "exec"), namespace)
    db = ParameterDatabase.get_default()
    terms = {
        "before": namespace["AtomTypeDependentTerm"](db, device),
        "after": AtomTypeDependentTerm(db, device),
    }
    pose = extended_pose_stack_from_sequences(["AA"], device=device)
    pbt = pose.packed_block_types
    for bt in pbt.active_block_types:
        terms["after"].setup_block_type(bt)

    def fresh():
        packed = copy.copy(pbt)
        for name in FIELDS:
            if hasattr(packed, name):
                delattr(packed, name)
        return packed

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    seconds = {k: [] for k in terms}
    examples = {}
    for repeat in range(8):
        for label in (list(terms) if repeat % 2 else list(terms)[::-1]):
            sets = [fresh() for _ in range(10)]
            synchronize()
            start = time.perf_counter()
            for packed in sets:
                terms[label].setup_packed_block_types(packed)
            synchronize()
            elapsed = (time.perf_counter() - start) / len(sets)
            if repeat:
                seconds[label].append(elapsed)
            examples[label] = packed
    for name in FIELDS:
        a, b = (getattr(examples[label], name) for label in terms)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b
    scalar_reads = {}
    for label, term in terms.items():
        packed = fresh()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            term.setup_packed_block_types(packed)
            synchronize()
        scalar_reads[label] = sum(
            event.count
            for event in profile.key_averages()
            if event.key == "aten::_local_scalar_dense"
        )
    result = dict(
        baseline=revision,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        device=str(device),
        torch=torch.__version__,
        n_types=pbt.n_types,
        n_real_atoms=sum(len(bt.atoms) for bt in pbt.active_block_types),
        seconds=seconds,
        medians={k: statistics.median(v) for k, v in seconds.items()},
        native_scalar_reads=scalar_reads,
        all_annotations_exact_match=True,
        scope="warm block metadata, fresh packed annotations; seven alternating sets of ten calls; separate instrumented scalar count; excludes pose creation, term construction and scoring",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
