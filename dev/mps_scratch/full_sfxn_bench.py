"""Full beta2016 energy: all-CPU vs ported-terms-on-MPS, with a cost breakdown.

usage: full_sfxn_bench.py [n_copies] [n_reps]
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES, build, _render  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402


def median(f, reps, sync):
    f()
    if sync:
        torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        if sync:
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    db = ParameterDatabase.get_default()
    cpu = torch.device("cpu")
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)

    cpu_stack, mps_stack, mix = build(one, n, db)
    ref = _render(cpu_stack, BETA_WEIGHTS, db, cpu)

    cc = cpu_stack.coords.detach().clone().requires_grad_(True)
    mc = mps_stack.coords.detach().clone().requires_grad_(True)

    def cpu_fwd():
        with torch.no_grad():
            ref(cpu_stack.coords)

    def mix_fwd():
        with torch.no_grad():
            mix(mps_stack.coords).sum()

    def cpu_fb():
        cc.grad = None
        ref(cc).sum().backward()

    def mix_fb():
        mc.grad = None
        mix(mc).sum().backward()

    # Cost attribution: the coordinate round trip the host terms force.
    host_coords = mps_stack.coords
    def xfer_down():
        host_coords.to("cpu")
    small = torch.zeros((), device=mps_stack.coords.device)
    def xfer_up():
        small.to("cpu")

    print(f"poses                  : {n} x 1UBQ ({one.coords.shape[1]} atoms/pose)")
    print(f"coords                 : {tuple(mps_stack.coords.shape)}")
    r = {}
    r["cpu_fwd"] = median(cpu_fwd, reps, False)
    r["mix_fwd"] = median(mix_fwd, reps, True)
    r["cpu_fb"] = median(cpu_fb, reps, False)
    r["mix_fb"] = median(mix_fb, reps, True)
    r["xfer_down"] = median(xfer_down, reps, True)
    r["xfer_up"] = median(xfer_up, reps, True)

    with torch.no_grad():
        e_cpu = float(ref(cpu_stack.coords).sum())
        e_mix = float(mix(mps_stack.coords).sum())
    print(f"energy cpu / mixed     : {e_cpu:.4f} / {e_mix:.4f}"
          f"  (rel {abs(e_cpu-e_mix)/abs(e_cpu):.2e})")
    print(f"forward      cpu       : {r['cpu_fwd']:8.2f} ms")
    print(f"forward      mixed     : {r['mix_fwd']:8.2f} ms   "
          f"{r['cpu_fwd']/r['mix_fwd']:.2f}x")
    print(f"fwd+bwd      cpu       : {r['cpu_fb']:8.2f} ms")
    print(f"fwd+bwd      mixed     : {r['mix_fb']:8.2f} ms   "
          f"{r['cpu_fb']/r['mix_fb']:.2f}x")
    print(f"coords mps->cpu        : {r['xfer_down']:8.2f} ms  "
          f"({100*r['xfer_down']/r['mix_fb']:.1f}% of mixed fwd+bwd)")
    print(f"scalar  mps->cpu       : {r['xfer_up']:8.2f} ms")
    print(f"terms on mps           : {', '.join(MPS_SCORE_TYPES)}")


if __name__ == "__main__":
    main()
