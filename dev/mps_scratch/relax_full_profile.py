"""Time all four operations of a relax cycle, on the host.

A relax cycle is not just minimization. It is:
  1. scoring all rotamers      -- the rotamer-pair energies feeding the graph
  2. SA over rotamer assignments
  3. minimization updates      -- the optimizer's own bookkeeping
  4. score + derivative calcs  -- the energy evaluations the optimizer drives

Only 3 and 4 have a Metal path today. This measures all four so the remaining
opportunity can be ranked by what it is actually worth.

usage: relax_full_profile.py [n_copies] [min_steps]
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import relax_steps as RS  # noqa: E402
from mixed_device import BETA_WEIGHTS  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.optimization._lbfgs_armijo import LBFGS_Armijo  # noqa: E402
from tmol.pack import PackerPalette, PackerTask, pack_rotamers  # noqa: E402
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler  # noqa: E402
from tmol.pack.rotamer.dunbrack import (  # noqa: E402
    create_dunbrack_sampler_from_database,
)
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    db = ParameterDatabase.get_default()
    dev = torch.device("cpu")
    torch.manual_seed(0)

    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), dev)
    stack = PoseStackBuilder.from_poses([one] * n, dev)

    sfxn = ScoreFunction(db, dev)
    for name, w in BETA_WEIGHTS.items():
        sfxn.set_weight(getattr(ScoreType, name), w)

    palette = PackerPalette()
    task = PackerTask(stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, dev))

    print(f"=== packing phase, {n} x 1UBQ (host) ===")
    t0 = time.perf_counter()
    packed = pack_rotamers(stack, sfxn, task, verbose=True)
    t_pack = time.perf_counter() - t0
    print(f"pack_rotamers total        {t_pack*1e3:9.1f} ms")

    print(f"\n=== minimization phase, {steps} iterations ===")
    net = RS.build(n, "cpu", db, ())
    opt = LBFGS_Armijo(
        net.parameters(),
        max_iter=steps,
        fixed_iterations=True,
        segment_ids=net.segment_ids,
    )
    evals = {"n": 0}

    def closure():
        evals["n"] += 1
        opt.zero_grad()
        e = net()
        e.sum().backward()
        return e

    net().sum().backward()
    net.zero_grad(set_to_none=True)
    evals["n"] = 0
    t0 = time.perf_counter()
    opt.step(closure)
    t_min = time.perf_counter() - t0
    print(f"minimization total         {t_min*1e3:9.1f} ms"
          f"  ({evals['n']} evaluations)")

    total = t_pack + t_min
    print(f"\n=== one relax cycle: {total*1e3:.1f} ms ===")
    print(f"  packing      {t_pack*1e3:9.1f} ms  {100*t_pack/total:5.1f}%")
    print(f"  minimization {t_min*1e3:9.1f} ms  {100*t_min/total:5.1f}%")
    _ = packed


if __name__ == "__main__":
    main()
