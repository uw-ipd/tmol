"""Run a beta2016 score function with the ported terms on MPS and the rest on the host.

Not every energy term has a Metal backend. A term without one cannot simply be
handed MPS coordinates: its pose-derived tensors come from the pose stack it
was set up against, and mixing an MPS tensor with a CPU one inside a single op
fails in the tensor-view layer. So the host terms are set up against a host
copy of the same pose stack, and only the coordinates cross the boundary --
once per call, in each direction, with autograd carrying the gradient back
through the same `.to()`.

This is a harness, not a library feature: productionizing it means teaching
ScoreFunction a per-term device, which is a design change. See MPS_PORT_NOTES.md.
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pose_to_device import to_device  # noqa: E402

from tmol.database import ParameterDatabase  # noqa: E402
from tmol.score import ScoreFunction  # noqa: E402
from tmol.score._score_types import ScoreType  # noqa: E402

# beta2016, as the .sfxn file spells it.
BETA_WEIGHTS = {
    "fa_ljatr": 1.0,
    "fa_ljrep": 0.55,
    "fa_lk": 1.0,
    "fa_elec": 1.0,
    "hbond": 1.0,
    "lk_ball_iso": -0.38,
    "lk_ball": 0.92,
    "lk_bridge": -0.33,
    "lk_bridge_uncpl": -0.33,
    "omega": 0.48,
    "rama": 1.0,
    "disulfide": 1.25,
    "metal_coordination": 1.0,
    "cart_lengths": 0.5,
    "cart_angles": 0.5,
    "cart_torsions": 0.5,
    "cart_impropers": 0.5,
    "cart_hxltorsions": 1.0,
    "dunbrack_rot": 0.76,
    "dunbrack_rotdev": 0.69,
    "dunbrack_semirot": 0.78,
    "ref": 1.0,
    "gen_torsions": 1.0,
}

# Score types whose forward *and* backward run on Metal.
#
# lk_ball is ported end to end now -- its tile shrank (0.88x -> 1.49x) and
# both of its backward kernels, the scoring derivatives and the water
# generator's, run on Metal. It is still left on the host here, because that
# is what measures faster for minimization:
#
#   100 poses, 20 fixed steps, per energy evaluation
#     host only                        229.7 ms
#     ljlk + elec on Metal              92.5 ms   <- this set
#     ljlk + elec + lk_ball on Metal   118.0 ms
#
# lk_ball's forward is 1.56x faster on Metal but its backward is 1.7x
# slower, so the term costs more on the GPU (50.2 ms) than on the host
# (43.4 ms) and moving it makes the GPU the longer pole. The backward is
# slow because its derivative kernels are arithmetic- and register-bound,
# not because of atomic contention -- Metal gives them 1.33x and 2.33x
# fewer resident threadgroups than their forward counterparts. See
# MPS_PORT_NOTES.md for the measurements and for one fix that was tried
# and reverted.
#
# The lk_ball port still pays off where no backward runs: rotamer-pair
# energies, where it is 2.6x (dev/mps_scratch/rpe_mps.py).
#
# hbond stays on the host for a different reason: see MPS_PORT_NOTES.md for
# the float32 coefficient error that kept it there.
# hbond is deliberately absent. Its whole-pose kernel is ported and correct,
# but it measures 0.78x against the host (see MPS_PORT_NOTES.md): the kernel
# is occupancy-bound on per-residue tile state rather than arithmetic, and
# adding it here lengthened minimization from 1143 ms to 1332 ms at 30 poses.
# Its rotamer kernel is a different story and does run on the device.
MPS_SCORE_TYPES = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")



def _render(stack, weights, db, device):
    if not weights:
        return None
    sfxn = ScoreFunction(db, device)
    for name, w in weights.items():
        sfxn.set_weight(getattr(ScoreType, name), w)
    return sfxn.render_whole_pose_scoring_module(stack)


class _QuantizeGrad(torch.autograd.Function):
    """Snap the gradient flowing back through here onto a fixed grid.

    Diagnostic only. The Metal gradient path accumulates through atomic float
    adds whose arrival order varies, so it is reproducible only to about
    7.6e-06. Rounding to a grid makes it deterministic while *degrading*
    accuracy, which separates the two: if a deterministic-but-coarser gradient
    restores the host's iteration count, order-dependence is what costs the
    extra evaluations, not accuracy.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return torch.round(g * ctx.scale) / ctx.scale, None


class _MetalValueHostGrad(torch.autograd.Function):
    """Diagnostic: take the energy from Metal and the gradient from the host.

    The Metal gradient is scattered into per-atom slots through atomic float
    adds, so it is neither reproducible nor a smooth function of the
    coordinates. Substituting the host's gradient while keeping the Metal
    energy isolates whether that is what costs the extra line-search
    evaluations. Deliberately slow -- it evaluates the host terms a second
    time -- and only for measurement.
    """

    @staticmethod
    def forward(ctx, coords, gpu_mod, host_mirror, host_coords):
        ctx.host_mirror = host_mirror
        ctx.host_coords = host_coords
        ctx.device = coords.device
        with torch.no_grad():
            return gpu_mod(coords)

    @staticmethod
    def backward(ctx, grad_out):
        with torch.enable_grad():
            c = ctx.host_coords.detach().clone().requires_grad_(True)
            out = ctx.host_mirror(c)
            (out * grad_out.detach().to("cpu")).sum().backward()
        return c.grad.to(ctx.device), None, None, None


class MixedDeviceScoring(torch.nn.Module):
    """Sum of a Metal-resident partial score and a host-resident one.

    Accepts coordinates on either device and moves them where each half needs
    them. Which side the caller's coordinates live on is not ours to choose:
    the kinematic layer has thirteen dispatched ops and none of them has a
    Metal backend, so a torsion-space relax keeps its coordinates on the host
    and this module copies up. A Cartesian caller already on MPS copies down.
    """

    def __init__(
        self,
        cpu_stack,
        mps_stack,
        db,
        mps_score_types=MPS_SCORE_TYPES,
        quantize_gpu_grad=None,
        host_grad_for_gpu_terms=False,
    ):
        super().__init__()
        self.quantize_gpu_grad = quantize_gpu_grad
        self.host_grad_for_gpu_terms = host_grad_for_gpu_terms
        gpu_w = {k: v for k, v in BETA_WEIGHTS.items() if k in mps_score_types}
        cpu_w = {k: v for k, v in BETA_WEIGHTS.items() if k not in mps_score_types}
        self.gpu_device = mps_stack.coords.device
        self.gpu_mod = _render(mps_stack, gpu_w, db, self.gpu_device)
        self.cpu_mod = _render(cpu_stack, cpu_w, db, torch.device("cpu"))
        # Host mirror of the Metal terms, for the diagnostic above only.
        self.gpu_mirror = (
            _render(cpu_stack, gpu_w, db, torch.device("cpu"))
            if host_grad_for_gpu_terms
            else None
        )

    def forward(self, coords):
        # Order matters. Metal work is enqueued asynchronously -- the whole
        # ljlk+elec forward and backward returns to Python in 0.2 ms and then
        # runs for ~50 ms on the GPU -- while a device-to-host copy blocks
        # until the queue drains. So the host copy is taken before anything is
        # enqueued: it then costs only its own bandwidth (0.2 ms for 100
        # poses) instead of serializing the two halves.
        host_coords = None
        if self.cpu_mod is not None:
            host_coords = coords if coords.device.type == "cpu" else coords.to("cpu")
        gpu_coords = None
        if self.gpu_mod is not None:
            gpu_coords = (
                coords
                if coords.device == self.gpu_device
                else coords.to(self.gpu_device)
            )

        # Per-pose energies, not a scalar: KinForestSfxnNetwork.forward is
        # expected to return one energy per pose, and run_min minimizes them
        # independently.
        total = None
        if self.gpu_mod is not None:
            if self.quantize_gpu_grad is not None:
                gpu_coords = _QuantizeGrad.apply(gpu_coords, self.quantize_gpu_grad)
            if self.gpu_mirror is not None:
                total = _MetalValueHostGrad.apply(
                    gpu_coords, self.gpu_mod, self.gpu_mirror, host_coords
                )
            else:
                total = self.gpu_mod(gpu_coords)
            if self.cpu_mod is not None:
                # Enqueuing is not submitting. Torch's MPS backend records work
                # into a command buffer and only commits it when something
                # forces it -- a sync, a copy, or a size threshold -- so
                # without this the GPU sits idle through the host half and
                # starts only when the final add waits on it. Recording an
                # event commits the buffer without waiting, which is what
                # turns the two halves from sequential into concurrent: the
                # add's wait drops from 49 ms to 13 ms at 100 poses.
                # torch.mps has no explicit commit(); an event is the handle.
                torch.mps.Event().record()
        if self.cpu_mod is not None:
            host = self.cpu_mod(host_coords)
            if total is None:
                return host.to(coords.device)
            total = total.to(coords.device) + host.to(coords.device)
        elif total is not None:
            total = total.to(coords.device)
        return total


def build(one_pose, n_copies, db=None):
    """Return (cpu_stack, mps_stack, MixedDeviceScoring) for n copies of one_pose."""
    from tmol.pose import PoseStackBuilder

    db = db or ParameterDatabase.get_default()
    cpu_stack = PoseStackBuilder.from_poses([one_pose] * n_copies, torch.device("cpu"))
    mps_stack = to_device(cpu_stack, torch.device("mps"))
    return cpu_stack, mps_stack, MixedDeviceScoring(cpu_stack, mps_stack, db)
