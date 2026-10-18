"""Phase 0h: leave-one-out marginal cost of each term in the real beta2016 sfxn."""
import sys, time, json, torch
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.tests.data import data_path

N, T, REP = int(sys.argv[1]), int(sys.argv[2]), 7
torch.set_num_threads(T)
device = torch.device("cpu")

BETA = {"fa_ljatr":1.0,"fa_ljrep":.55,"fa_lk":1.0,"fa_elec":1.0,"hbond":1.0,
        "lk_ball_iso":-.38,"lk_ball":.92,"lk_bridge":-.33,"lk_bridge_uncpl":-.33,
        "omega":.48,"rama":1.0,"disulfide":1.25,"metal_coordination":1.0,
        "cart_lengths":.5,"cart_angles":.5,"cart_torsions":.5,"cart_impropers":.5,
        "cart_hxltorsions":1.0,"dunbrack_rot":.76,"dunbrack_rotdev":.69,
        "dunbrack_semirot":.78,"ref":1.0,"gen_torsions":1.0}
GROUPS = {
    "ljlk":       ["fa_ljatr","fa_ljrep","fa_lk"],
    "elec":       ["fa_elec"],
    "hbond":      ["hbond"],
    "lk_ball":    ["lk_ball_iso","lk_ball","lk_bridge","lk_bridge_uncpl"],
    "cartbonded": ["cart_lengths","cart_angles","cart_torsions","cart_impropers","cart_hxltorsions"],
    "bb_torsion": ["omega","rama"],
    "dunbrack":   ["dunbrack_rot","dunbrack_rotdev","dunbrack_semirot"],
    "genbonded":  ["gen_torsions"],
}
db = ParameterDatabase.get_default()
one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
stack = PoseStackBuilder.from_poses([one] * N, device)

def timeit(weights):
    sfxn = ScoreFunction(db, device)
    for st, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, st), w)
    wpsm = sfxn.render_whole_pose_scoring_module(stack)
    c = stack.coords.detach().clone().requires_grad_(True)
    for _ in range(2):
        wpsm(c).sum().backward(); c.grad = None
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        wpsm(c).sum().backward(); c.grad = None
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2] * 1000          # median

out = {"FULL": timeit(BETA)}
for name, sts in GROUPS.items():
    out[name] = timeit({k: v for k, v in BETA.items() if k not in sts})
print(json.dumps(out))
