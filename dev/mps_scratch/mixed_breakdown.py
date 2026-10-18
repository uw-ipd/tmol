"""Where the mixed-device score call spends its time, by term group.

Times the host half and the Metal half of the composite separately, then
leaves each host term group out in turn to get its marginal cost. That tells
us whether the remaining cost is the untouched host terms, the GPU kernels, or
the boundary between them.

usage: mixed_breakdown.py [n_copies] [n_reps]
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES, _render  # noqa: E402
from pose_to_device import to_device  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402

GROUPS = {
    "hbond": ["hbond"],
    "lk_ball": ["lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"],
    "cartbonded": [
        "cart_lengths",
        "cart_angles",
        "cart_torsions",
        "cart_impropers",
        "cart_hxltorsions",
    ],
    "bb_torsion": ["omega", "rama"],
    "dunbrack": ["dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"],
    "genbonded": ["gen_torsions"],
    "disulfide": ["disulfide"],
    "metal": ["metal_coordination"],
    "ref": ["ref"],
}


def median(fn, reps, sync):
    fn()
    if sync:
        torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
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
    mps = torch.device("mps")
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    cpu_stack = PoseStackBuilder.from_poses([one] * n, cpu)
    mps_stack = to_device(cpu_stack, mps)

    host_w = {k: v for k, v in BETA_WEIGHTS.items() if k not in MPS_SCORE_TYPES}
    gpu_w = {k: v for k, v in BETA_WEIGHTS.items() if k in MPS_SCORE_TYPES}

    def fb(mod, stack, sync):
        c = stack.coords.detach().clone().requires_grad_(True)

        def run():
            c.grad = None
            mod(c).sum().backward()

        return median(run, reps, sync)

    gpu_mod = _render(mps_stack, gpu_w, db, mps)
    host_mod = _render(cpu_stack, host_w, db, cpu)
    t_gpu = fb(gpu_mod, mps_stack, True)
    t_host = fb(host_mod, cpu_stack, False)

    # the same four terms, on the host, for the comparison
    gpu_on_cpu = _render(cpu_stack, gpu_w, db, cpu)
    t_gpu_on_cpu = fb(gpu_on_cpu, cpu_stack, False)

    print(f"poses                         : {n} x 1UBQ")
    print(f"ljlk+elec  on mps  (fwd+bwd)  : {t_gpu:8.2f} ms")
    print(f"ljlk+elec  on cpu  (fwd+bwd)  : {t_gpu_on_cpu:8.2f} ms"
          f"   -> {t_gpu_on_cpu/t_gpu:.2f}x")
    print(f"host terms (19)    (fwd+bwd)  : {t_host:8.2f} ms")
    print(f"sum of halves                 : {t_gpu + t_host:8.2f} ms")
    print(f"all-cpu equivalent            : {t_gpu_on_cpu + t_host:8.2f} ms")
    print()
    print("marginal cost of each host term group (leave-one-out, fwd+bwd):")
    for name, sts in GROUPS.items():
        kept = {k: v for k, v in host_w.items() if k not in sts}
        if not kept:
            continue
        mod = _render(cpu_stack, kept, db, cpu)
        t = fb(mod, cpu_stack, False)
        print(f"  {name:12s} {t_host - t:7.2f} ms   "
              f"({100*(t_host-t)/(t_gpu+t_host):5.1f}% of mixed total)")


if __name__ == "__main__":
    main()
