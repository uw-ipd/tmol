"""LJ/LK latency and agreement with block-pair sums across batch sizes."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
import tmol
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_cif
from tmol.pose import PoseStackBuilder
from tmol.score.ljlk import LJLKEnergyTerm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    database = ParameterDatabase.get_default()
    one = pose_stack_from_cif(
        Path(tmol.__file__).parent / "tests/data/cif/1UBQ.cif", device, no_optH=True
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for batch in (1, 4, 16):
        pose = PoseStackBuilder.from_poses([one] * batch, device)
        term = LJLKEnergyTerm(database, device)
        for bt in pose.packed_block_types.active_block_types:
            term.setup_block_type(bt)
        term.setup_packed_block_types(pose.packed_block_types)
        term.setup_poses(pose)
        whole = term.render_whole_pose_scoring_module(pose)
        pairs = term.render_block_pair_scoring_module(pose)
        with torch.no_grad():
            expected = pairs(pose.coords).double().sum(dim=(-1, -2))
            measured = whole(pose.coords)
            error = float((measured.double() - expected).abs().max())
            durations = []
            for repetition in range(22):
                synchronize()
                start = time.perf_counter()
                whole(pose.coords)
                synchronize()
                if repetition >= 2:
                    durations.append(time.perf_counter() - start)
        row = dict(
            batch=batch,
            blocks=pose.max_n_blocks,
            seconds=durations,
            median=statistics.median(durations),
            max_error_vs_precise_pair_sum=error,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        diff=subprocess.check_output(
            ["git", "diff", "--", "tmol/score/ljlk"], text=True
        ),
        torch=torch.__version__,
        device=str(device),
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
