"""Phase 0j: separate the two CPU parallelism layers.
W = concurrent score terms (python ThreadPoolExecutor)
T = intra-op OpenMP threads per term (torch.set_num_threads)
Reports wall time and average cores busy (CPU time / wall time)."""
import sys, time, json, resource, torch
import tmol.score._score_function as SF
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.tests.data import data_path

N, W, T, REP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 7
SF._MAX_CPU_SCORE_TERM_WORKERS = W
torch.set_num_threads(T)
device = torch.device("cpu")
BETA = {"fa_ljatr":1.0,"fa_ljrep":.55,"fa_lk":1.0,"fa_elec":1.0,"hbond":1.0,
        "lk_ball_iso":-.38,"lk_ball":.92,"lk_bridge":-.33,"lk_bridge_uncpl":-.33,
        "omega":.48,"rama":1.0,"disulfide":1.25,"metal_coordination":1.0,
        "cart_lengths":.5,"cart_angles":.5,"cart_torsions":.5,"cart_impropers":.5,
        "cart_hxltorsions":1.0,"dunbrack_rot":.76,"dunbrack_rotdev":.69,
        "dunbrack_semirot":.78,"ref":1.0,"gen_torsions":1.0}
db = ParameterDatabase.get_default()
one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
stack = PoseStackBuilder.from_poses([one]*N, device)
sfxn = ScoreFunction(db, device)
for st, v in BETA.items(): sfxn.set_weight(getattr(ScoreType, st), v)
wpsm = sfxn.render_whole_pose_scoring_module(stack)
c = stack.coords.detach().clone().requires_grad_(True)
for _ in range(3): wpsm(c).sum().backward(); c.grad = None

def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime

ts = []
c0, w0 = cpu_s(), time.perf_counter()
for _ in range(REP):
    t0 = time.perf_counter(); wpsm(c).sum().backward(); c.grad = None
    ts.append(time.perf_counter()-t0)
cpu, wall = cpu_s()-c0, time.perf_counter()-w0
ts.sort()
print(json.dumps({"W":W,"T":T,"ms":ts[len(ts)//2]*1000,"cores_busy":cpu/wall}))
