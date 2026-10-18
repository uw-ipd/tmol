"""Isolated elec fwd+bwd timing, median of 9."""
import sys, time, torch
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.tests.data import data_path

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
device = torch.device("cpu")
one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
stack = PoseStackBuilder.from_poses([one] * int(sys.argv[1]), device)
sfxn = ScoreFunction(ParameterDatabase.get_default(), device)
sfxn.set_weight(ScoreType.fa_elec, 1.0)
wpsm = sfxn.render_whole_pose_scoring_module(stack)
c = stack.coords.detach().clone().requires_grad_(True)
for _ in range(3):
    wpsm(c).sum().backward(); c.grad = None
ts = []
for _ in range(9):
    t0 = time.perf_counter(); wpsm(c).sum().backward(); c.grad = None
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"elec fwd+bwd  median {ts[4]*1000:7.1f} ms   min {ts[0]*1000:7.1f} ms")
