"""Which energy terms dominate the rotamer-pair energy calculation.

calcRPEs is the largest phase of a relax cycle, so what it is made of decides
where a Metal port is worth writing. Runs the real packing path with subsets
of the score function and reports the calcRPEs stage of each.

usage: rpe_terms.py [n_copies]
"""
import io
import re
import sys
from contextlib import redirect_stdout

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mixed_device import BETA_WEIGHTS  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.io import pose_stack_from_pdb  # noqa: E402
from tmol.pack import PackerPalette, PackerTask, pack_rotamers  # noqa: E402
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler  # noqa: E402
from tmol.pack.rotamer.dunbrack import (  # noqa: E402
    create_dunbrack_sampler_from_database,
)
from tmol.pose import PoseStackBuilder  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402
from tmol.tests.data import data_path  # noqa: E402

LJLK_ELEC = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")


def run(stack, task, weights, db, dev):
    sfxn = ScoreFunction(db, dev)
    for name, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, name), w)
    buf = io.StringIO()
    with redirect_stdout(buf):
        pack_rotamers(stack, sfxn, task, verbose=True)
    out = buf.getvalue()
    tot = sum(float(x) for x in re.findall(r"calcRPEs:\s+([\d.]+)", out))
    whole = sum(float(x) for x in re.findall(r"pack_rotamers\s+([\d.]+)", out))
    return tot, whole


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    db = ParameterDatabase.get_default()
    dev = torch.device("cpu")
    torch.manual_seed(0)
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), dev)
    stack = PoseStackBuilder.from_poses([one] * n, dev)
    palette = PackerPalette()
    task = PackerTask(stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, dev))

    LK_BALL = ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl")
    HBOND = ("hbond",)
    PORTED = LJLK_ELEC + LK_BALL
    cases = {
        "full beta2016": BETA_WEIGHTS,
        "ljlk + elec only": {k: v for k, v in BETA_WEIGHTS.items() if k in LJLK_ELEC},
        "lk_ball only": {k: v for k, v in BETA_WEIGHTS.items() if k in LK_BALL},
        "hbond only": {k: v for k, v in BETA_WEIGHTS.items() if k in HBOND},
        "ljlk+elec+lk_ball": {k: v for k, v in BETA_WEIGHTS.items() if k in PORTED},
        "all but those three": {
            k: v for k, v in BETA_WEIGHTS.items() if k not in PORTED
        },
        # Leave-one-out against the full score function. Terms share rotamer
        # dispatch indices when their cutoffs are compatible, so a term's
        # standalone cost includes infrastructure it would not pay for on its
        # own in a real run. The marginal number is the honest one.
        "full minus lk_ball": {
            k: v for k, v in BETA_WEIGHTS.items() if k not in LK_BALL
        },
        "full minus ljlk+elec": {
            k: v for k, v in BETA_WEIGHTS.items() if k not in LJLK_ELEC
        },
        "full minus hbond": {
            k: v for k, v in BETA_WEIGHTS.items() if k not in HBOND
        },
    }
    print(f"rotamer-pair energies, {n} x 1UBQ (host)")
    for name, w in cases.items():
        rpe, whole = run(stack, task, w, db, dev)
        print(f"  {name:20s} calcRPEs {rpe*1e3:9.1f} ms"
              f"   pack total {whole*1e3:9.1f} ms")


if __name__ == "__main__":
    main()
