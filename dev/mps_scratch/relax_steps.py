"""Fixed-step torsion relax: equal work on every backend, so timings compare.

A full relax runs until convergence, so its wall time mixes two things -- how
fast an evaluation is and how many the line search needs. Both matter, but
only the first is a performance question. This harness runs a fixed number of
optimizer steps instead, so every backend does the same work.

usage: relax_steps.py [n_copies] [mode] [n_steps] [mps_score_types]
  mode: cpu | mps | composite-cpu
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES, MixedDeviceScoring  # noqa: E402
from pose_to_device import to_device  # noqa: E402

from tmol import FoldForest, MoveMap  # noqa: E402
from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.optimization import build_kinforest_network  # noqa: E402
from tmol.optimization._lbfgs_armijo import LBFGS_Armijo  # noqa: E402
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402


def build(n, mode, db, types):
    cpu = torch.device("cpu")
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    cpu_stack = PoseStackBuilder.from_poses([one] * n, cpu)
    mixed = None
    if mode == "cpu":
        stack = cpu_stack
        sfxn = ScoreFunction(db, cpu)
        for name, w in BETA_WEIGHTS.items():
            sfxn.set_weight(getattr(ScoreType, name), w)
    else:
        # The kinematic layer has no Metal backend, so coordinates stay on the
        # host and the composite copies up for the Metal half.
        mps_stack = (
            cpu_stack if mode == "composite-cpu"
            else to_device(cpu_stack, torch.device("mps"))
        )
        stack = cpu_stack
        sfxn = ScoreFunction(db, cpu)
        for name in types:
            sfxn.set_weight(getattr(ScoreType, name), BETA_WEIGHTS[name])
        mixed = MixedDeviceScoring(
            cpu_stack, mps_stack, db, mps_score_types=types
        )

    ff = FoldForest.reasonable_fold_forest(stack)
    mm = MoveMap.from_pose_stack(stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    net = build_kinforest_network(stack, sfxn, ff, mm, False)
    if mixed is not None:
        net.whole_pose_scoring_module = mixed
    return net


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "mps"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    types = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else MPS_SCORE_TYPES
    db = ParameterDatabase.get_default()
    torch.manual_seed(0)

    net = build(n, mode, db, types)
    sync = mode == "mps"
    evals = {"n": 0}

    def closure():
        evals["n"] += 1
        opt.zero_grad()
        e = net()
        e.sum().backward()
        return e

    # fixed_iterations keeps the optimizer from stopping early, so every
    # backend performs the same number of iterations and the wall times are
    # comparable. The evaluation count can still differ -- the line search
    # inside an iteration is free to take more or fewer trial steps -- so it
    # is reported alongside.
    opt = LBFGS_Armijo(
        net.parameters(),
        max_iter=steps,
        fixed_iterations=True,
        segment_ids=net.segment_ids,
    )

    # Warm up: the first call compiles shaders and allocates.
    net().sum().backward()
    net.zero_grad(set_to_none=True)
    if sync:
        torch.mps.synchronize()

    evals["n"] = 0
    t0 = time.perf_counter()
    opt.step(closure)
    if sync:
        torch.mps.synchronize()
    wall = time.perf_counter() - t0

    e = float(net().sum())
    print(f"mode {mode:14s} poses {n:4d}  steps {steps:3d}")
    print(f"  wall                 : {wall:8.3f} s")
    print(f"  per optimizer step   : {wall / steps * 1e3:8.2f} ms")
    print(f"  energy evaluations   : {evals['n']:5d}"
          f"  ({wall / max(evals['n'], 1) * 1e3:.2f} ms each)")
    print(f"  E after              : {e:.4f}")


if __name__ == "__main__":
    main()
