"""Validate and time rotamer-pair energies on MPS against the host.

Builds one rotamer set on the host, scores it with a single-term score
function on each device, and compares the per-pair energies the packer would
consume. Usage:

    python dev/mps_scratch/rpe_mps.py [term] [n_copies]

term is one of hbond, elec, ljlk, lk_ball, all.
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pose_to_device import to_device  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pack import PackerPalette, PackerTask  # noqa: E402
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler  # noqa: E402
from tmol.pack.rotamer.dunbrack import (  # noqa: E402
    create_dunbrack_sampler_from_database,
)
from tmol.pack import SetPackerTask  # noqa: E402
from tmol.pack.rotamer import build_rotamers  # noqa: E402
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


def _move(obj, device):
    """Shallow-move every tensor attribute of an attrs record to `device`."""
    import attr

    if not attr.has(type(obj)):
        return obj
    changes = {}
    for field in attr.fields(type(obj)):
        v = getattr(obj, field.name)
        if isinstance(v, torch.Tensor):
            changes[field.name] = v.to(device)
        elif attr.has(type(v)):
            changes[field.name] = _move(v, device)
    return attr.evolve(obj, **changes) if changes else obj


def score(stack, rotamer_set, weights, db, device):
    sfxn = ScoreFunction(db, device)
    for name, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, name), w)
    module = sfxn.render_rotamer_scoring_module(stack, rotamer_set)
    return module(rotamer_set.coords)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    db = ParameterDatabase.get_default()
    cpu = torch.device("cpu")
    mps = torch.device("mps")
    torch.manual_seed(0)

    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), cpu)
    stack = PoseStackBuilder.from_poses([one] * n, cpu)
    palette = PackerPalette()
    task = PackerTask(stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, cpu))
    stack, rotamer_set = build_rotamers(
        stack, SetPackerTask.from_packer_task(task), stack.packed_block_types.chem_db
    )
    print(f"poses {n} x 1UBQ, {rotamer_set.coords.shape[0]} rotamer atoms")

    mps_stack = to_device(stack, mps)
    mps_rots = _move(rotamer_set, mps)

    names = list(TERMS) if which == "all" else [which]
    for name in names:
        w = TERMS[name]
        ref = score(stack, rotamer_set, w, db, cpu).coalesce()
        got = score(mps_stack, mps_rots, w, db, mps).cpu().coalesce()
        n = max(int(ref.shape[1]), int(ref.shape[2]))
        # A rotamer-pair cell is identified by (pose, rot1, rot2); linearize so
        # the two sparse layouts can be matched without a Python loop.
        ka = ((ref.indices()[0] * n) + ref.indices()[1]) * n + ref.indices()[2]
        kb = ((got.indices()[0] * n) + got.indices()[1]) * n + got.indices()[2]
        va, vb = ref.values().double(), got.values().double()
        sa, ia = torch.sort(ka)
        sb, ib = torch.sort(kb)
        va, vb = va[ia], vb[ib]
        pos = torch.searchsorted(sb, sa).clamp(max=sb.numel() - 1)
        hit = sb[pos] == sa
        err = (va[hit] - vb[pos[hit]]).abs().max().item()
        rel = err / max(va.abs().max().item(), 1e-12)
        only_a = int((~hit).sum())
        posb = torch.searchsorted(sa, sb).clamp(max=sa.numel() - 1)
        hitb = sa[posb] == sb
        only_b = int((~hitb).sum())
        extra = 0.0
        if only_a:
            extra = max(extra, va[~hit].abs().max().item())
        if only_b:
            extra = max(extra, vb[~hitb].abs().max().item())
        print(
            f"  {name:8s} shared {int(hit.sum()):8d}  max abs {err:.3e} "
            f"rel {rel:.3e}   cpu-only {only_a} mps-only {only_b} "
            f"(largest {extra:.3e})"
        )
        print(
            f"           pose totals: cpu {va.sum().item():.6f} "
            f"mps {vb.sum().item():.6f}"
        )

        for dev, st, rs in ((cpu, stack, rotamer_set), (mps, mps_stack, mps_rots)):
            ts = []
            for _ in range(5):
                torch.mps.synchronize() if dev.type == "mps" else None
                t0 = time.perf_counter()
                score(st, rs, w, db, dev)
                torch.mps.synchronize() if dev.type == "mps" else None
                ts.append(time.perf_counter() - t0)
            ts.sort()
            print(f"           {dev.type:4s} median {ts[len(ts)//2]*1e3:8.1f} ms")


if __name__ == "__main__":
    main()
