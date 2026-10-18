"""Validate and time the LK-ball term on MPS against the CPU backend.

Scores N copies of 1UBQ on each device, compares the energies, then times the
forward pass. Usage:

    python dev/mps_scratch/lk_ball_mps_bench.py [n_copies] [n_reps]
"""

import sys
import time

import torch

from tmol.database import ParameterDatabase
from tmol.pose import PoseStackBuilder
from tmol.score.lk_ball import LKBallEnergyTerm
from tmol.tests.data.pdb import data as pdb_data
from tmol.io import pose_stack_from_pdb

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pose_to_device import to_device  # noqa: E402


def prepare(n_copies, device, db):
    # Built on CPU and relocated: pose construction runs operators that have
    # no Metal backend yet, so only the scoring call is on the GPU.
    cpu = torch.device("cpu")
    one = pose_stack_from_pdb(pdb_data["1UBQ"], cpu)
    poses = PoseStackBuilder.from_poses([one] * n_copies, cpu)
    if device.type != "cpu":
        poses = to_device(poses, device)

    term = LKBallEnergyTerm(param_db=db, device=device)
    for bt in poses.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(poses.packed_block_types)
    term.setup_poses(poses)
    return poses, term.render_whole_pose_scoring_module(poses)


def main():
    n_copies = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    db = ParameterDatabase.get_default()

    out = {}
    for name in ("cpu", "mps"):
        device = torch.device(name)
        poses, module = prepare(n_copies, device, db)

        value = module(poses.coords)
        if name == "mps":
            torch.mps.synchronize()
        out[name] = float(value.sum())

        times = []
        for _ in range(n_reps):
            start = time.perf_counter()
            module(poses.coords)
            if name == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
        times.sort()
        out[name + "_ms"] = times[len(times) // 2] * 1e3

    print(f"poses      : {n_copies} x 1UBQ")
    print(f"cpu  energy: {out['cpu']:.4f}")
    print(f"mps  energy: {out['mps']:.4f}")
    delta = abs(out["cpu"] - out["mps"])
    print(f"abs/rel err: {delta:.4e} / {delta / max(abs(out['cpu']), 1e-9):.3e}")
    print(f"cpu  median: {out['cpu_ms']:.2f} ms")
    print(f"mps  median: {out['mps_ms']:.2f} ms")
    print(f"speedup    : {out['cpu_ms'] / out['mps_ms']:.2f}x")


if __name__ == "__main__":
    main()
