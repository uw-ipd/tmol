"""Paired timing of old chemical-group inference and explicit cached correlation.

Only the mask lookup stage is timed. Real joint-sampler outputs must match;
independent OptH outputs intentionally have different correlation semantics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pose._conjugated_groups import lockstep_group_for_block
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _pose, _task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    revision = "77f5d94190f4c358fc4cad13b9e97d975c446fb8"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:tmol/pose/_conjugated_groups.py"], text=True
    )
    namespace = {}
    exec(compile(source, "baseline_conjugated_groups.py", "exec"), namespace)
    old = namespace["lockstep_group_for_block"]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for fixture, stem in FIXTURES.items():
        pose, context = _pose(stem, device)
        task, _ = _task(pose, context.parameter_database, device)
        task.set_chi_sample_budget(4096, 2048)
        pose, rotamers = build_rotamers(
            pose,
            SetPackerTask.from_packer_task(task),
            context.parameter_database.chemical,
        )
        torch.testing.assert_close(
            old(pose, rotamers), lockstep_group_for_block(pose, rotamers)
        )
        seconds = {"before": [], "after": []}
        for repeat in range(8):
            methods = [("before", old), ("after", lockstep_group_for_block)]
            for label, method in methods[:: 1 if repeat % 2 else -1]:
                synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    method(pose, rotamers)
                synchronize()
                if repeat:
                    seconds[label].append((time.perf_counter() - start) / 100)
        before_masks = [old(pose, rotamers) for _ in range(10)]
        after_masks = [lockstep_group_for_block(pose, rotamers) for _ in range(10)]
        sizes = {
            label: sum(
                {m.data_ptr(): m.numel() * m.element_size() for m in masks}.values()
            )
            for label, masks in (("before", before_masks), ("after", after_masks))
        }
        row = dict(
            fixture=fixture,
            n_blocks=pose.max_n_blocks,
            n_rotamers=rotamers.n_rotamers_total,
            seconds=seconds,
            medians={
                label: statistics.median(values) for label, values in seconds.items()
            },
            retained_mask_storage_bytes_for_10_consumers=sizes,
            joint_mask_exact_match=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=revision,
                baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                device=str(device),
                torch=torch.__version__,
                rows=rows,
                scope="warm mask lookup only; seven alternating-order sets of 100 calls; not full render or scoring",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
