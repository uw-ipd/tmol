"""Validate one term's whole-pose score and gradient on MPS against the host.

Scores the same pose stack with a single-term score function on each device and
compares the pose totals and the per-atom coordinate gradients, then times the
forward-plus-backward pass on each.

usage: pose_check.py [term] [n_copies]
  term is a key of TERMS below, or "all".
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pose_to_device import to_device  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402

TERMS = {
    "hbond": {"hbond": 1.0},
    "elec": {"fa_elec": 1.0},
    "ljlk": {"fa_ljatr": 1.0, "fa_ljrep": 0.55, "fa_lk": 1.0},
    "lk_ball": {
        "lk_ball_iso": -0.38,
        "lk_ball": 0.92,
        "lk_bridge": -0.33,
        "lk_bridge_uncpl": -0.33,
    },
}


def score_and_grad(stack, weights, db, device):
    sfxn = ScoreFunction(db, device)
    for name, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, name), w)
    module = sfxn.render_whole_pose_scoring_module(stack)
    coords = stack.coords.clone().detach().requires_grad_(True)
    total = module(coords)
    total.sum().backward()
    return total.detach(), coords.grad.detach()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    db = ParameterDatabase.get_default()
    cpu, mps = torch.device("cpu"), torch.device("mps")
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    cpu_stack = PoseStackBuilder.from_poses([one] * n, cpu)
    mps_stack = to_device(cpu_stack, mps)
    print(f"poses {n} x 1UBQ, {cpu_stack.coords.shape[1]} atoms per pose")

    for name in list(TERMS) if which == "all" else [which]:
        w = TERMS[name]
        e_ref, g_ref = score_and_grad(cpu_stack, w, db, cpu)
        e_got, g_got = score_and_grad(mps_stack, w, db, mps)
        e_got, g_got = e_got.cpu().double(), g_got.cpu().double()
        e_ref, g_ref = e_ref.double(), g_ref.double()
        de = (e_ref - e_got).abs().max().item()
        scale = max(e_ref.abs().max().item(), 1e-12)
        # Compare the gradient where it is actually resolvable: a per-atom
        # component far below float32 precision of the pose total carries no
        # information, and dividing by it manufactures huge relative errors.
        gmag = g_ref.abs().max().item()
        live = g_ref.abs() > 1e-4 * gmag
        dg = (g_ref - g_got).abs()
        dg_max = dg.max().item()
        dg_rel = (dg[live] / g_ref[live].abs()).max().item() if live.any() else 0.0
        print(
            f"  {name:8s} energy  ref {e_ref.sum().item():14.6f} "
            f"mps {e_got.sum().item():14.6f}  "
            f"max abs {de:.3e} rel {de / scale:.3e}"
        )
        print(
            f"           gradient  max abs {dg_max:.3e}  "
            f"max rel (|g| > 1e-4 peak) {dg_rel:.3e}  "
            f"peak |g| {gmag:.3e}"
        )

        row = []
        for dev, st in ((cpu, cpu_stack), (mps, mps_stack)):
            ts = []
            for _ in range(5):
                if dev.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                score_and_grad(st, w, db, dev)
                if dev.type == "mps":
                    torch.mps.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            row.append(ts[len(ts) // 2] * 1e3)
        print(
            f"           fwd+bwd   cpu {row[0]:8.1f} ms   mps {row[1]:8.1f} ms"
            f"   {row[0] / row[1]:.2f}x"
        )


if __name__ == "__main__":
    main()
