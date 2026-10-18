"""Per-term timing of both halves of a relax cycle, host and MPS.

Times every term of the beta score function twice: once as rotamer-pair
energies (what calcRPEs does) and once as a whole-pose forward+backward (what
each minimization evaluation does). Terms with a Metal kernel are timed on both
devices; the rest are host-only and say so.

usage: term_breakdown.py [n_copies] [which]
  which: rpe | pose | both   (default both)
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES  # noqa: E402
from pose_to_device import to_device  # noqa: E402
from relax_mixed_full import MPS_RPE_TYPES, _move_rotamer_set  # noqa: E402

# Every term with a Metal kernel, whether or not the mixed split currently
# routes it there. MPS_SCORE_TYPES is the split; this is the capability.
PORTED_POSE = (
    "fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec", "hbond",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
)

# Terms that share one kernel, so their individual timings must not be summed.
KERNEL_GROUPS = {
    "ljlk": ("fa_ljatr", "fa_ljrep", "fa_lk"),
    "lk_ball": ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"),
    "cartbonded": ("cart_lengths", "cart_angles", "cart_torsions",
                   "cart_impropers", "cart_hxltorsions"),
    "dunbrack": ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
}

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pack import PackerPalette, PackerTask, SetPackerTask  # noqa: E402
from tmol.pack.rotamer import (  # noqa: E402
    FixedAAChiSampler,
    IncludeCurrentSampler,
    build_rotamers,
)
from tmol.pack.rotamer.dunbrack import (  # noqa: E402
    create_dunbrack_sampler_from_database,
)
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402

CPU, MPS = torch.device("cpu"), torch.device("mps")


def _sfxn(db, device, name):
    sfxn = ScoreFunction(db, device)
    sfxn.set_weight(getattr(ScoreType, name), BETA_WEIGHTS[name])
    return sfxn


def _median(fn, n=3):
    ts = []
    for _ in range(n):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def time_rpe(stack, rots, db, device, name):
    sfxn = _sfxn(db, device, name)
    module = sfxn.render_rotamer_scoring_module(stack, rots)
    coords = rots.coords
    module(coords)  # warm
    return _median(lambda: module(coords))


def time_pose(stack, db, device, name):
    sfxn = _sfxn(db, device, name)
    module = sfxn.render_whole_pose_scoring_module(stack)
    coords = stack.coords.clone().detach().requires_grad_(True)

    def once():
        if coords.grad is not None:
            coords.grad = None
        module(coords).sum().backward()

    once()  # warm
    return _median(once)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    db = ParameterDatabase.get_default()
    torch.manual_seed(0)
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), CPU)
    stack = PoseStackBuilder.from_poses([one] * n, CPU)

    if which in ("pose", "both"):
        mps_stack = to_device(stack, MPS)
        print(f"\nwhole-pose forward+backward, {n} x 1UBQ "
              f"(one minimization evaluation)")
        print(f"  {'term':22s} {'host':>10s} {'mps':>10s} {'speedup':>9s}")
        seen = set()
        for name in BETA_WEIGHTS:
            group = next((g for g, members in KERNEL_GROUPS.items()
                          if name in members), None)
            if group is not None:
                if group in seen:
                    continue
                seen.add(group)
            h = time_pose(stack, db, CPU, name)
            if name in PORTED_POSE:
                m = time_pose(mps_stack, db, MPS, name)
                label = (group or name)
                print(f"  {label:22s} {h:9.1f}ms {m:9.1f}ms {h / m:8.2f}x")
            else:
                label = (group or name)
                print(f"  {label:22s} {h:9.1f}ms {'host only':>10s}")

    if which in ("rpe", "both"):
        palette = PackerPalette()
        task = PackerTask(stack, palette)
        task.restrict_to_repacking()
        task.add_conformer_sampler(IncludeCurrentSampler())
        task.add_conformer_sampler(FixedAAChiSampler())
        task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, CPU))
        rot_stack, rots = build_rotamers(
            stack, SetPackerTask.from_packer_task(task),
            stack.packed_block_types.chem_db)
        mps_rot_stack = to_device(rot_stack, MPS)
        mps_rots = _move_rotamer_set(rots, MPS)
        print(f"\nrotamer-pair energies, {n} x 1UBQ, "
              f"{rots.coords.shape[0]} rotamer atoms (calcRPEs)")
        print(f"  {'term':22s} {'host':>10s} {'mps':>10s} {'speedup':>9s}")
        seen = set()
        for name in BETA_WEIGHTS:
            group = next((g for g, members in KERNEL_GROUPS.items()
                          if name in members), None)
            if group is not None:
                if group in seen:
                    continue
                seen.add(group)
            h = time_rpe(rot_stack, rots, db, CPU, name)
            if name in MPS_RPE_TYPES:
                m = time_rpe(mps_rot_stack, mps_rots, db, MPS, name)
                label = (group or name)
                print(f"  {label:22s} {h:9.1f}ms {m:9.1f}ms {h / m:8.2f}x")
            else:
                label = (group or name)
                print(f"  {label:22s} {h:9.1f}ms {'host only':>10s}")


if __name__ == "__main__":
    main()
