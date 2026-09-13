"""Compare anchor-only library sampling with the pre-fix implementation."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from tmol.io import pose_stack_from_cif
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
from tmol.pack.rotamer._conjugated_groups import (
    add_conjugated_group_sampler,
    find_conjugated_groups,
)
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    baseline_commit = "c2a9b4c1c"
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{baseline_commit}:tmol/pack/rotamer/_conjugated_chi_sampler.py",
        ],
        text=True,
    )
    namespace = {}
    exec(compile(source, "baseline_conjugated_chi_sampler.py", "exec"), namespace)
    baseline = namespace["ConjugatedChiSampler"].anchor_library_chi
    candidate = ConjugatedChiSampler.anchor_library_chi
    rows = []
    for fixture in ("lys_biotin_1bdo", "oglycan_sia_1g1s", "nglycan_tree_1ax2"):
        pose, context = pose_stack_from_cif(
            Path("tmol/tests/data/covalent_fixtures") / f"{fixture}.cif",
            device,
            prepare_ligands=True,
            ligand_seed=20250828,
            no_optH=True,
            return_context=True,
        )
        task = PackerTask(pose, PackerPalette())
        library = create_dunbrack_sampler_from_database(
            context.parameter_database, device
        )
        task.add_conformer_sampler(library)
        sampler = add_conjugated_group_sampler(task, pose)
        task.restrict_to_repacking()
        task = SetPackerTask.from_packer_task(task)
        for rt in pose.packed_block_types.active_block_types:
            library.annotate_residue_type(rt)
        library.annotate_packed_block_types(pose.packed_block_types)
        groups = find_conjugated_groups(pose)
        original_mask = task.per_block_conformer_sampler_allowed.clone()
        sample = type(library).sample_chi_for_poses
        counts = []

        def counted_sample(self, *a, **kw):
            result = sample(self, *a, **kw)
            counts.append(result[1].numel())
            return result

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        times = {"before": [], "after": []}
        outputs, sampled_rows = {}, {}
        with patch.object(type(library), "sample_chi_for_poses", counted_sample):
            for repeat in range(args.repeats + 1):
                # Alternate order to reduce systematic warm/order bias.
                methods = [("before", baseline), ("after", candidate)]
                for label, method in methods[:: 1 if repeat % 2 == 0 else -1]:
                    synchronize()
                    start = time.perf_counter()
                    outputs[label] = method(sampler, pose, task, groups)
                    synchronize()
                    elapsed = time.perf_counter() - start
                    sampled_rows[label] = counts[-1]
                    torch.testing.assert_close(
                        task.per_block_conformer_sampler_allowed, original_mask
                    )
                    if repeat:
                        times[label].append(elapsed)
                assert outputs["before"].keys() == outputs["after"].keys()
                for key in outputs["before"]:
                    for old, new in zip(outputs["before"][key], outputs["after"][key]):
                        np.testing.assert_array_equal(old, new)
        row = dict(
            fixture=fixture,
            n_groups=len(groups),
            library_rows=sampled_rows,
            seconds=times,
            medians={k: statistics.median(v) for k, v in times.items()},
            exact_anchor_output_match=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        baseline_commit=baseline_commit,
        candidate_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        candidate_diff=subprocess.check_output(
            ["git", "diff", "--", "tmol/pack/rotamer/_conjugated_chi_sampler.py"],
            text=True,
        ),
        device=str(device),
        torch=torch.__version__,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
