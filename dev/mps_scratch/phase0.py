"""Phase 0: time-breakdown of a torsion-space relax of N copies of 1UBQ (CPU)."""
import cProfile, pstats, io as _io, sys, time
import torch

from tmol import PoseStack, beta2016_score_function, FoldForest, MoveMap
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.optimization import build_kinforest_network, run_min
from tmol.tests.data import data_path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
device = torch.device("cpu")
torch.manual_seed(0)

t0 = time.perf_counter()
one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
print(f"load 1ubq:            {time.perf_counter()-t0:7.3f}s  "
      f"({one.coords.shape[1]} atoms/pose)")

t0 = time.perf_counter()
stack = PoseStackBuilder.from_poses([one] * N, device)
print(f"build stack of {N:3d}:   {time.perf_counter()-t0:7.3f}s  "
      f"coords={tuple(stack.coords.shape)}")

sfxn = beta2016_score_function(device)
ff = FoldForest.reasonable_fold_forest(stack)
mm = MoveMap.from_pose_stack(stack)
mm.move_all_jumps = True
mm.move_all_named_torsions = True

t0 = time.perf_counter()
net = build_kinforest_network(stack, sfxn, ff, mm, False)
print(f"build kinforest net:  {time.perf_counter()-t0:7.3f}s")

# ---- one forward/backward, timed in isolation -------------------------------
def fwdbwd():
    net.zero_grad(set_to_none=True)
    e = net().sum()
    e.backward()
    return e

fwdbwd()  # warm up
t0 = time.perf_counter(); e = fwdbwd(); t_fb = time.perf_counter() - t0
print(f"1 fwd+bwd:            {t_fb:7.3f}s   E={float(e):.3f}")

# ---- profiled minimization --------------------------------------------------
print(f"\n--- cProfile: full torsion relax, {N} x 1UBQ ---")
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
run_min(net)
pr.disable()
wall = time.perf_counter() - t0
print(f"total minimization wall: {wall:.3f}s\n")

s = _io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(45)
print(s.getvalue())
