"""Paired warm setup of new packed sets sharing one generic parameter database."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.io import extended_pose_stack_from_sequences
from tmol.database import ParameterDatabase
from tmol.score.genbonded import GenBondedEnergyTerm
from tmol.score.genbonded._genbonded_energy_term import _PACKED_FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    revision = "d0d19b405e0aaeedd4c5ad68c7948850ede87f3b"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:tmol/score/genbonded/_genbonded_energy_term.py"],
        text=True,
    )
    namespace = {}
    exec(compile(source, "baseline_generic_setup.py", "exec"), namespace)
    database = ParameterDatabase.get_default()
    before = namespace["GenBondedEnergyTerm"](database, device)
    after = GenBondedEnergyTerm(database, device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for reuse_common in (False, True):
        pose = extended_pose_stack_from_sequences(["KK"], device=device)
        pbt = pose.packed_block_types
        for bt in pbt.active_block_types:
            after.setup_block_type(bt)
        if reuse_common:
            # Separate generic setup from the shared parent annotation cost.
            after.setup_packed_block_types(pbt)

        def fresh():
            result = copy.copy(pbt)
            for name in (*_PACKED_FIELDS, "_genbonded_parameters"):
                if hasattr(result, name):
                    delattr(result, name)
            return result

        seconds = {"before": [], "after": []}
        examples = {}
        for repeat in range(8):
            methods = [("before", before), ("after", after)]
            for label, term in methods[:: 1 if repeat % 2 else -1]:
                sets = [fresh() for _ in range(10)]
                synchronize()
                start = time.perf_counter()
                for packed in sets:
                    term.setup_packed_block_types(packed)
                synchronize()
                elapsed = (time.perf_counter() - start) / len(sets)
                if repeat:
                    seconds[label].append(elapsed)
                examples[label] = sets
        for name in _PACKED_FIELDS:
            torch.testing.assert_close(
                getattr(examples["before"][0], name),
                getattr(examples["after"][0], name),
                rtol=0,
                atol=0,
            )
        storage = {}
        for label, sets in examples.items():
            tensors = [
                getattr(packed, name) for packed in sets for name in _PACKED_FIELDS
            ]
            storage[label] = sum(
                {t.data_ptr(): t.numel() * t.element_size() for t in tensors}.values()
            )
        row = dict(
            reuse_common_annotations=reuse_common,
            n_types=pbt.n_types,
            seconds=seconds,
            medians={k: statistics.median(v) for k, v in seconds.items()},
            retained_tensor_bytes_for_10_packed_sets=storage,
            all_annotation_tensors_exact_match=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=revision,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                device=str(device),
                torch=torch.__version__,
                rows=rows,
                scope="warm packed-set setup, shared immutable block types/database; excludes pose creation and scorer rendering; tensor storage excludes float64 scoring copies and allocator overhead",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
