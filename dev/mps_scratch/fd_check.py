"""How small a line-search step each backend can still resolve.

LBFGS_Armijo validates its analytic directional derivative against a finite
difference of the energy. This measures the agreement of the two as the step
shrinks, which is the quantity that decides whether a trial step is accepted.

usage: fd_check.py [n_copies]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES, _render  # noqa: E402
from pose_to_device import to_device  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    db = ParameterDatabase.get_default()
    cpu = torch.device("cpu")
    mps = torch.device("mps")
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    cpu_stack = PoseStackBuilder.from_poses([one] * n, cpu)
    mps_stack = to_device(cpu_stack, mps)
    gw = {k: v for k, v in BETA_WEIGHTS.items() if k in MPS_SCORE_TYPES}

    torch.manual_seed(0)
    direction_cpu = torch.randn_like(cpu_stack.coords)
    direction_cpu /= direction_cpu.norm()

    steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    print(f"poses {n} x 1UBQ;  ljlk+elec only;  random unit direction")
    print(f"{'step':>8}  {'cpu rel err':>12}  {'mps rel err':>12}")
    rows = {}
    for dev, stack, name in ((cpu, cpu_stack, "cpu"), (mps, mps_stack, "mps")):
        mod = _render(stack, gw, db, dev)
        d = direction_cpu.to(dev)
        c = stack.coords.detach().clone().requires_grad_(True)
        e0 = mod(c).sum()
        e0.backward()
        if dev.type == "mps":
            torch.mps.synchronize()
        analytic = float((c.grad * d).sum())
        base = float(e0)
        errs = []
        for s in steps:
            with torch.no_grad():
                e1 = float(mod(stack.coords + s * d).sum())
            if dev.type == "mps":
                torch.mps.synchronize()
            fd = (e1 - base) / s
            errs.append(abs(fd - analytic) / max(abs(analytic), 1e-30))
        rows[name] = errs
        print(f"  (analytic directional derivative, {name}: {analytic:.4f})")
    for i, s in enumerate(steps):
        print(f"{s:8.0e}  {rows['cpu'][i]:12.3e}  {rows['mps'][i]:12.3e}")


if __name__ == "__main__":
    main()
