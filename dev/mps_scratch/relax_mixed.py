"""Torsion-space relax of N x 1UBQ with the ported terms on MPS.

Builds the kinematic network against the MPS pose stack, then swaps its
whole-pose scoring module for the mixed-device composite. The network only
uses the score function to render that module, so the substitution is the
whole integration.

usage: relax_mixed.py [n_copies] [cpu|mps]
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MixedDeviceScoring, _render  # noqa: E402
from pose_to_device import to_device  # noqa: E402

from tmol import FoldForest, MoveMap  # noqa: E402
from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.optimization import build_kinforest_network, run_min  # noqa: E402
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "mps"
    db = ParameterDatabase.get_default()
    cpu = torch.device("cpu")
    torch.manual_seed(0)

    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    cpu_stack = PoseStackBuilder.from_poses([one] * n, cpu)

    if mode == "composite-cpu":
        # Control: the same two-half composite, both halves on the host. Isolates
        # whether the extra line-search evaluations come from the Metal backend
        # or from the composite's structure -- two separately weighted partial
        # scores summed per pose, rather than one fused score function.
        stack = cpu_stack
        sfxn = ScoreFunction(db, cpu)
        for name in ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"):
            sfxn.set_weight(getattr(ScoreType, name), BETA_WEIGHTS[name])
        mixed = MixedDeviceScoring(cpu_stack, cpu_stack, db)
    elif mode == "cpu":
        stack = cpu_stack
        sfxn = ScoreFunction(db, cpu)
        for name, w in BETA_WEIGHTS.items():
            sfxn.set_weight(getattr(ScoreType, name), w)
        mixed = None
    else:
        # The kinematic layer stays on the host: none of its thirteen
        # dispatched ops has a Metal backend, so the network's coordinates are
        # CPU-resident and the composite copies up for the Metal half.
        mps_stack = to_device(cpu_stack, torch.device("mps"))
        stack = cpu_stack
        sfxn = ScoreFunction(db, cpu)
        for name in ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"):
            sfxn.set_weight(getattr(ScoreType, name), BETA_WEIGHTS[name])
        # Optional third argument: comma-separated score types to put on Metal,
        # for bisecting which kernel a behaviour comes from.
        from mixed_device import MPS_SCORE_TYPES as DEFAULT_MPS
        types = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT_MPS
        mixed = MixedDeviceScoring(
            cpu_stack,
            mps_stack,
            db,
            mps_score_types=types,
            host_grad_for_gpu_terms=(
                len(sys.argv) > 4 and sys.argv[4].startswith("hostgrad")
            ),
            quantize_gpu_grad=(
                float(sys.argv[5]) if len(sys.argv) > 5 else None
            ),
        )

    ff = FoldForest.reasonable_fold_forest(stack)
    mm = MoveMap.from_pose_stack(stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    t0 = time.perf_counter()
    net = build_kinforest_network(stack, sfxn, ff, mm, False)
    t_build = time.perf_counter() - t0
    if mixed is not None:
        net.whole_pose_scoring_module = mixed

    def fwdbwd():
        net.zero_grad(set_to_none=True)
        e = net().sum()
        e.backward()
        if mode != "cpu":
            torch.mps.synchronize()
        return e

    e0 = float(fwdbwd())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        fwdbwd()
        ts.append(time.perf_counter() - t)
    ts.sort()

    t0 = time.perf_counter()
    run_min(net)
    if mode != "cpu":
        torch.mps.synchronize()
    t_min = time.perf_counter() - t0

    e1 = float(net().sum())
    print(f"mode                 : {mode}")
    print(f"poses                : {n} x 1UBQ")
    print(f"build kinforest net  : {t_build:8.3f} s")
    print(f"1 fwd+bwd (median)   : {ts[len(ts)//2]*1e3:8.2f} ms")
    print(f"full relax           : {t_min:8.3f} s")
    print(f"E start -> end       : {e0:.3f} -> {e1:.3f}")


if __name__ == "__main__":
    main()
