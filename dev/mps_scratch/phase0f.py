"""Phase 0f: per-term CPU thread scaling, each term isolated in its own ScoreFunction."""
import sys, time, json
import torch
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.tests.data import data_path

N, NTHREADS, REP = int(sys.argv[1]), int(sys.argv[2]), 5
torch.set_num_threads(NTHREADS)
device = torch.device("cpu")

TERMS = {
    "cartbonded":  {"cart_lengths":.5,"cart_angles":.5,"cart_torsions":.5,
                    "cart_impropers":.5,"cart_hxltorsions":1.0},
    "hbond":       {"hbond":1.0},
    "elec":        {"fa_elec":1.0},
    "ljlk":        {"fa_ljatr":1.0,"fa_ljrep":.55,"fa_lk":1.0},
    "lk_ball":     {"lk_ball_iso":-.38,"lk_ball":.92,"lk_bridge":-.33,"lk_bridge_uncpl":-.33},
    "bb_torsion":  {"omega":.48,"rama":1.0},
    "genbonded":   {"gen_torsions":1.0},
    "dunbrack":    {"dunbrack_rot":.76,"dunbrack_rotdev":.69,"dunbrack_semirot":.78},
}

db = ParameterDatabase.get_default()
one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
stack = PoseStackBuilder.from_poses([one] * N, device)

out = {}
for name, weights in TERMS.items():
    sfxn = ScoreFunction(db, device)
    for st, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, st), w)
    wpsm = sfxn.render_whole_pose_scoring_module(stack)
    c = stack.coords.detach().clone().requires_grad_(True)
    wpsm(c).sum().backward(); c.grad = None          # warm up
    t0 = time.perf_counter()
    for _ in range(REP):
        wpsm(c).sum().backward(); c.grad = None
    out[name] = (time.perf_counter() - t0) / REP * 1000
print(json.dumps(out))
