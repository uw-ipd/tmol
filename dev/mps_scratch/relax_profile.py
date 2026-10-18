"""Where a fixed-step torsion relax spends its time, by phase.

Separates the optimizer's own work from the energy evaluations it drives, and
splits an evaluation into the kinematic mapping and the scoring halves. Every
number is wall time on the host; the Metal half is synchronized where noted.

usage: relax_profile.py [n_copies] [steps]
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import relax_steps as RS  # noqa: E402
from mixed_device import MPS_SCORE_TYPES  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.optimization._lbfgs_armijo import LBFGS_Armijo  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    db = ParameterDatabase.get_default()
    torch.manual_seed(0)
    net = RS.build(n, "mps", db, MPS_SCORE_TYPES)
    mix = net.whole_pose_scoring_module

    acc = {k: 0.0 for k in ("kin_fwd", "score", "net_bwd", "eval")}
    counts = {"eval": 0}

    orig_forward = type(net).forward

    def timed_forward(self):
        counts["eval"] += 1
        t0 = time.perf_counter()
        self.full_dofs = self.full_dofs.detach()
        self.full_coords = self.full_coords.detach()
        self.flat_coords = self.flat_coords.detach()
        self.full_dofs.view(-1)[self._dof_flat_idx] = self.masked_dofs
        kin_coords = self.kin_module(self.full_dofs)
        self.flat_coords[self.id[1:]] = kin_coords[1:].to(self.flat_coords.dtype)
        self.full_coords = self.flat_coords.view(self.orig_coords_shape)
        t1 = time.perf_counter()
        out = self.whole_pose_scoring_module(self.full_coords)
        t2 = time.perf_counter()
        acc["kin_fwd"] += t1 - t0
        acc["score"] += t2 - t1
        return out

    type(net).forward = timed_forward

    def closure():
        opt.zero_grad()
        e = net()
        t0 = time.perf_counter()
        e.sum().backward()
        torch.mps.synchronize()
        acc["net_bwd"] += time.perf_counter() - t0
        return e

    opt = LBFGS_Armijo(
        net.parameters(),
        max_iter=steps,
        fixed_iterations=True,
        segment_ids=net.segment_ids,
    )
    net().sum().backward()
    net.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    for k in acc:
        acc[k] = 0.0
    counts["eval"] = 0

    t0 = time.perf_counter()
    opt.step(closure)
    torch.mps.synchronize()
    wall = time.perf_counter() - t0
    type(net).forward = orig_forward

    ev = counts["eval"]
    inside = acc["kin_fwd"] + acc["score"] + acc["net_bwd"]
    print(f"poses {n}, {steps} optimizer iterations, {ev} energy evaluations")
    print(f"  total wall                 {wall*1e3:9.1f} ms")
    print(f"  kinematics forward         {acc['kin_fwd']*1e3:9.1f} ms"
          f"  {100*acc['kin_fwd']/wall:5.1f}%")
    print(f"  scoring forward            {acc['score']*1e3:9.1f} ms"
          f"  {100*acc['score']/wall:5.1f}%")
    print(f"  backward (kin + scoring)   {acc['net_bwd']*1e3:9.1f} ms"
          f"  {100*acc['net_bwd']/wall:5.1f}%")
    print(f"  optimizer itself           {(wall-inside)*1e3:9.1f} ms"
          f"  {100*(wall-inside)/wall:5.1f}%")
    print(f"  per evaluation             {wall/ev*1e3:9.2f} ms")


if __name__ == "__main__":
    main()
