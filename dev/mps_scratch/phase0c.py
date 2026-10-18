"""Phase 0c: per-term forward/backward cost of static scoring, N x 1UBQ, CPU."""
import sys, time, cProfile, pstats, io as _io
import torch
from tmol import beta2016_score_function
from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.tests.data import data_path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REP = 10
device = torch.device("cpu")

one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), device)
stack = PoseStackBuilder.from_poses([one] * N, device)
sfxn = beta2016_score_function(device)
wpsm = sfxn.render_whole_pose_scoring_module(stack)
coords = stack.coords.detach().clone().requires_grad_(True)

wpsm(coords).sum().backward(); coords.grad = None   # warm up

pr = cProfile.Profile(); pr.enable()
for _ in range(REP):
    wpsm(coords).sum().backward()
    coords.grad = None
pr.disable()

s = _io.StringIO()
st = pstats.Stats(pr, stream=s)
rows = []
for (fn, ln, name), (cc, nc, tt, ct, _) in st.stats.items():
    if "torch._ops.tmol" in name or "tmol_" in name:
        rows.append((tt / REP, name.replace("built-in method torch._ops.", ""), nc // REP))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"{N} x 1UBQ, fwd+bwd scoring, mean of {REP} reps  "
      f"(threads={torch.get_num_threads()})\n")
print(f"{'native op':<44}{'ms/call':>10}{'calls':>7}{'% kernels':>11}")
for t, name, c in rows:
    print(f"{name:<44}{t*1000:>10.1f}{c:>7}{100*t/tot:>10.1f}%")
print(f"{'TOTAL kernel self-time':<44}{tot*1000:>10.1f}")
