"""A whole relax cycle with the ported terms on Metal, stage by stage.

A relax cycle is four operations, not just minimization:

  1. scoring all rotamers      -- the rotamer-pair energies feeding the graph
  2. building the interaction graph
  3. SA over rotamer assignments
  4. minimization: score + derivative calcs the optimizer drives

Only 1 and 4 have Metal kernels. The packer's own machinery does not: the
rotamer builder, the interaction graph and the annealer are host-only, and
PackerTask cannot even be constructed against an MPS pose stack. So the
orchestration stays on the host and stage 1 is hooked at its one entry point,
``ScoreFunction.render_rotamer_scoring_module`` -- the same split
MixedDeviceScoring applies to whole-pose scoring.

usage: relax_mixed_full.py [n_copies] [min_steps] [mode]
  mode: host | mixed   (default: run both and compare)
"""

import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import relax_steps as RS  # noqa: E402
from mixed_device import BETA_WEIGHTS, MPS_SCORE_TYPES, MixedDeviceScoring  # noqa: E402

# The rotamer-pair split is not the minimization split. lk_ball loses in the
# minimization composite -- it lengthens the GPU pole past the host's -- but
# its rotamer kernel is ported and there is no backward in packing, so it
# belongs on the device here. hbond is the same: its rotamer kernel and its
# derived-atom pre-pass both have Metal entry points, but its pose-score
# kernel does not yet.
MPS_RPE_TYPES = MPS_SCORE_TYPES + (
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
    "hbond",
)
from pose_to_device import to_device  # noqa: E402

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

MPS = torch.device("mps")
CPU = torch.device("cpu")


def _move_rotamer_set(rotamer_set, device):
    """Shallow-move every tensor field of the rotamer set record."""
    import attr

    changes = {}
    for field in attr.fields(type(rotamer_set)):
        v = getattr(rotamer_set, field.name)
        if isinstance(v, torch.Tensor):
            changes[field.name] = v.to(device)
    return attr.evolve(rotamer_set, **changes)


def _render(stack, rotamer_set, weights, db, device):
    if not weights:
        return None
    sfxn = ScoreFunction(db, device)
    for name, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, name), w)
    return sfxn.render_rotamer_scoring_module(stack, rotamer_set)


class MixedRotamerScoring:
    """Rotamer-pair energies split across two devices, summed on the host.

    The packer consumes one sparse COO table of per-rotamer-pair energies. The
    ported terms build theirs on Metal and the rest on the host; sparse
    addition merges the two layouts, and the caller coalesces.
    """

    def __init__(self, cpu_stack, cpu_rots, db, mps_types):
        host_weights = {k: v for k, v in BETA_WEIGHTS.items() if k not in mps_types}
        gpu_weights = {k: v for k, v in BETA_WEIGHTS.items() if k in mps_types}
        self.host = _render(cpu_stack, cpu_rots, host_weights, db, CPU)
        mps_stack = to_device(cpu_stack, MPS)
        mps_rots = _move_rotamer_set(cpu_rots, MPS)
        self.gpu = _render(mps_stack, mps_rots, gpu_weights, db, MPS)
        # The sparse table's shape, taken from any rendered term.
        probe = self.gpu if self.gpu is not None else self.host
        term = probe.term_modules[0]
        self.shape = (term.n_poses, term.n_rots, term.n_rots)

    def __call__(self, coords):
        parts = []
        gpu_result = None
        if self.gpu is not None:
            # Launch, but do not wait: issue the device half, run the host
            # half, and only then pull the device result back.
            gpu_result = self.gpu(coords.to(MPS))
            # Enqueuing is not the same as starting. PyTorch's MPS backend
            # holds the command buffer open until something waits on it, so
            # without this the queued kernels did not begin until the collect
            # below and the host half overlapped nothing. Recording an event
            # commits the buffer without waiting on it. At 100 poses this
            # takes calcRPEs from 21830 ms to 16740 -- the collect falls from
            # 10130 ms to 4996 while the host half is unchanged, which is the
            # device half running underneath it.
            torch.mps.Event().record()
        if self.host is not None:
            # __call__ rather than forward_sparse_entries: on the host it runs
            # _try_coalesce_cpu_rotamer_layouts, which merges duplicate
            # layouts before the packer's own pass. Bypassing it to save the
            # int64 promotion measured slower -- 4220 ms against 3180 at 10
            # poses -- because the packer then coalesces more entries.
            parts.append(self.host(coords))
        if gpu_result is not None:
            # Now wait, with the host half already done.
            parts.insert(0, gpu_result.cpu())
        if len(parts) == 1:
            return parts[0]
        # Concatenate rather than add. Sparse addition coalesces, and the
        # caller coalesces again straight afterwards; one pass over the
        # entries is enough, and duplicates across the two halves are exactly
        # what coalescing is for.
        indices = torch.cat([p._indices() for p in parts], dim=1)
        values = torch.cat([p._values() for p in parts])
        return torch.sparse_coo_tensor(indices, values, parts[0].shape)



def _stages(output):
    """Pull the per-stage timings out of pack_rotamers' verbose line."""
    pat = (
        r"pack_rotamers\s+([\d.]+) build rots:\s+([\d.]+) calcRPEs:\s+([\d.]+)"
        r" build IG:\s+([\d.]+) build IG part2:\s+([\d.]+)"
        r" run SA:\s+([\d.]+) pose ctor:\s+([\d.]+)"
    )
    acc = {}
    for m in re.finditer(pat, output):
        vals = [float(x) for x in m.groups()]
        for key, v in zip(
            ("total", "build rots", "calcRPEs", "build IG", "build IG2", "run SA",
             "pose ctor"), vals):
            acc[key] = acc.get(key, 0.0) + v
    return acc


def run(n, steps, mixed, db):
    torch.manual_seed(0)
    one = pose_stack_from_pdb(data_path("pdb/1ubq.pdb"), CPU)
    stack = PoseStackBuilder.from_poses([one] * n, CPU)

    sfxn = ScoreFunction(db, CPU)
    for name, w in BETA_WEIGHTS.items():
        sfxn.set_weight(getattr(ScoreType, name), w)

    palette = PackerPalette()
    task = PackerTask(stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(db, CPU))

    if mixed:
        # The one hook the packing path needs; everything else stays host-side.
        sfxn.render_rotamer_scoring_module = (
            lambda ps, rs: MixedRotamerScoring(ps, rs, db, MPS_RPE_TYPES)
        )

    buf = StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        pack_rotamers(stack, sfxn, task, verbose=True)
    t_pack = time.perf_counter() - t0
    stages = _stages(buf.getvalue())

    net = RS.build(n, "mps" if mixed else "cpu", db, MPS_SCORE_TYPES)
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
    if mixed:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    opt.step(closure)
    if mixed:
        torch.mps.synchronize()
    t_min = time.perf_counter() - t0
    return t_pack, t_min, stages, evals["n"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    db = ParameterDatabase.get_default()

    modes = {"host": [False], "mixed": [True]}.get(which, [False, True])
    out = {}
    for mixed in modes:
        out["mixed" if mixed else "host"] = run(n, steps, mixed, db)

    print(f"\none relax cycle, {n} x 1UBQ, {steps} fixed minimization steps")
    print(f"{'stage':18s} " + " ".join(f"{k:>12s}" for k in out))
    keys = ["build rots", "calcRPEs", "build IG", "build IG2", "run SA", "pose ctor"]
    for k in keys:
        row = " ".join(
            f"{out[m][2].get(k, 0.0) * 1e3:12.1f}" for m in out
        )
        print(f"  {k:16s} {row}")
    for label, idx in (("packing total", 0), ("minimization", 1)):
        row = " ".join(f"{out[m][idx] * 1e3:12.1f}" for m in out)
        print(f"  {label:16s} {row}")
    row = " ".join(f"{(out[m][0] + out[m][1]) * 1e3:12.1f}" for m in out)
    print(f"  {'CYCLE':16s} {row}")
    if len(out) == 2:
        h = out["host"][0] + out["host"][1]
        m = out["mixed"][0] + out["mixed"][1]
        print(f"\n  cycle speedup: {h / m:.2f}x"
              f"   (evaluations: host {out['host'][3]}, mixed {out['mixed'][3]})")


if __name__ == "__main__":
    main()
