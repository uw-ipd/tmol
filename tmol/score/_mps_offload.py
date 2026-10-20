"""Run a CPU score function's Metal-capable terms on the Apple GPU.

On Apple silicon the poses, the packer and the minimizer stay on the host --
they have no Metal kernels -- but the pairwise terms that dominate scoring do.
A CPU ``ScoreFunction`` therefore renders those terms against a Metal copy of
the poses and the rest on the host, and sums the two halves. Memory is
unified, so moving coordinates across costs little, and the halves run
concurrently: the Metal work is committed before the host half starts.

Placement is per term and per use, because a Metal kernel is not always the
faster one. Whole-pose scoring offloads LJ/LK and electrostatics only: lk_ball's
backward and hbond's pose kernel measure slower on Metal than on the host.
Rotamer-pair scoring has no backward, and there lk_ball and hbond win too.

Set ``TMOL_MPS_OFFLOAD=0`` to keep everything on the host.
"""

import os

import attr
import torch

from tmol.score._score_types import ScoreType

WHOLE_POSE_SCORE_TYPES = frozenset(
    (ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk, ScoreType.fa_elec)
)
ROTAMER_SCORE_TYPES = WHOLE_POSE_SCORE_TYPES | frozenset(
    (
        ScoreType.lk_ball_iso,
        ScoreType.lk_ball,
        ScoreType.lk_bridge,
        ScoreType.lk_bridge_uncpl,
        ScoreType.hbond,
    )
)

_MPS = torch.device("mps")


def mps_offload_enabled() -> bool:
    """Whether CPU scoring should hand its Metal-capable terms to the GPU."""
    if os.environ.get("TMOL_MPS_OFFLOAD", "1").lower() in ("0", "false", "off"):
        return False
    if not torch.backends.mps.is_available():
        return False
    from tmol._cpp_lib import has_mps

    return has_mps()


def to_mps(obj, _memo=None):
    """Copy a PoseStack, RotamerSet or similar attrs record onto the GPU.

    Residue types are shared rather than copied: the energy terms annotate
    them in place, and PackedBlockTypes already carries their device tables.
    Derived annotations on PackedBlockTypes are not attrs fields, so the copy
    drops them and the Metal score function rebuilds its own.
    """
    _memo = {} if _memo is None else _memo
    if id(obj) in _memo:
        return _memo[id(obj)]
    if isinstance(obj, torch.Tensor):
        moved = obj.to(_MPS)
    elif isinstance(obj, (list, tuple)):
        moved = type(obj)(to_mps(v, _memo) for v in obj)
    elif isinstance(obj, dict):
        moved = {k: to_mps(v, _memo) for k, v in obj.items()}
    elif type(obj).__name__ in ("RefinedResidueType", "ResidueType"):
        moved = obj
    elif attr.has(type(obj)):
        changes = {}
        for field in attr.fields(type(obj)):
            value = getattr(obj, field.name)
            new = _MPS if field.name == "device" else to_mps(value, _memo)
            if new is not value:
                changes[field.name] = new
        moved = attr.evolve(obj, **changes) if changes else obj
    else:
        moved = obj
    _memo[id(obj)] = moved
    return moved


def _commit_mps():
    """Start the queued Metal work without waiting for it.

    PyTorch holds the MPS command buffer open until something forces it, so
    without this the GPU half would only start once its result is read and
    would overlap nothing. Recording an event commits the buffer.
    """
    torch.mps.Event().record()


@attr.s(auto_attribs=True, frozen=True)
class OffloadPlan:
    """Which of a score function's terms, and weight lanes, go to the GPU."""

    host_terms: tuple
    gpu_terms: tuple
    host_lanes: torch.Tensor
    gpu_lanes: torch.Tensor

    @classmethod
    def for_terms(cls, terms, offloadable):
        host_terms, gpu_terms, host_lanes, gpu_lanes = [], [], [], []
        lane = 0
        for term in terms:
            n = len(term.score_types())
            on_gpu = all(st in offloadable for st in term.score_types())
            (gpu_terms if on_gpu else host_terms).append(term)
            (gpu_lanes if on_gpu else host_lanes).extend(range(lane, lane + n))
            lane += n
        if not gpu_terms:
            return None
        return cls(
            tuple(host_terms),
            tuple(gpu_terms),
            torch.tensor(host_lanes, dtype=torch.int64),
            torch.tensor(gpu_lanes, dtype=torch.int64),
        )

    def unsplit_lanes(self, host, gpu):
        """Reassemble per-lane results into the score function's lane order."""
        if host is None:
            return gpu
        order = torch.argsort(torch.cat([self.host_lanes, self.gpu_lanes]))
        return torch.cat([host, gpu])[order.to(host.device)]


class OffloadedWholePoseScoringModule:
    """A WholePoseScoringModule split between the host and the GPU.

    ``weights`` keeps the full, host-resident lane order, so callers that
    refresh weights in place (as the minimizers do between relax stages)
    see the same interface as the unsplit module.

    Metal has no float64, so coordinates of any other dtype are scored
    entirely on the host, by a host rendering of the offloaded terms that
    ``render_host_twin`` builds the first time one is needed.
    """

    def __init__(self, plan, weights, host_module, gpu_module, render_host_twin):
        self.plan = plan
        self.weights = torch.nn.Parameter(weights.unsqueeze(1), requires_grad=False)
        self.host = host_module
        self.gpu = gpu_module
        self._render_host_twin = render_host_twin
        self._host_twin = None
        self.term_modules = (
            *(host_module.term_modules if host_module else ()),
            *gpu_module.term_modules,
        )
        self._synced_weights_version = None

    def _sync_weights(self):
        version = self.weights._version
        if version == self._synced_weights_version:
            return
        with torch.no_grad():
            if self.host is not None:
                self.host.weights.copy_(self.weights[self.plan.host_lanes])
            self.gpu.weights.copy_(self.weights[self.plan.gpu_lanes])
            if self._host_twin is not None:
                self._host_twin.weights.copy_(self.weights[self.plan.gpu_lanes])
        self._synced_weights_version = version

    def _run(self, coords, call_gpu, call_host):
        if coords.dtype != torch.float32:
            if self._host_twin is None:
                self._host_twin = self._render_host_twin()
                self._synced_weights_version = None
            self._sync_weights()
            host = call_host(self.host, coords) if self.host is not None else None
            return host, call_host(self._host_twin, coords)
        self._sync_weights()
        gpu = call_gpu(self.gpu, coords.to(_MPS))
        _commit_mps()
        host = call_host(self.host, coords) if self.host is not None else None
        return host, gpu.to(coords.device)

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        def call(module, c):
            return module(c, sum_terms=sum_terms, apply_weights=apply_weights)

        host, gpu = self._run(coords, call, call)
        if sum_terms:
            return gpu if host is None else host + gpu
        return self.plan.unsplit_lanes(host, gpu)

    def unweighted_scores(self, coords, *, fused_score_weights=None):
        def call(module, c):
            return module.unweighted_scores(c)

        return self.plan.unsplit_lanes(*self._run(coords, call, call))


class OffloadedRotamerScoringModule:
    """A RotamerScoringModule split between the host and the GPU.

    The packer's CPU interaction-graph builder accumulates without atomics, so
    it runs in parallel only over entries with distinct (pose, rot1, rot2)
    keys and serially over the rest. ``unique_prefix_nnz`` tells it how many
    leading entries are distinct. The GPU terms share one rotamer-pair
    dispatch, which visits each pair once, so when the GPU half comes back as
    a single layout it is distinct throughout. It goes first, and it is
    nearly the whole table: the host terms only add entries at pairs it
    already holds. The host half, coalesced and small, is the tail.
    """

    def __init__(self, host_module, gpu_module):
        self.host = host_module
        self.gpu = gpu_module
        self.term_modules = (
            *(host_module.term_modules if host_module else ()),
            *gpu_module.term_modules,
        )
        term = gpu_module.term_modules[0]
        self.shape = (term.n_poses, term.n_rots, term.n_rots)
        self.unique_prefix_nnz = None

    def forward_sparse_entries(self, coords):
        """Return int32 ``(indices, values)``, GPU entries first.

        Sets ``unique_prefix_nnz`` for the table returned.
        """
        gpu_indices, gpu_values, _, _ = self.gpu._weighted_entries_by_layout(
            coords.to(_MPS)
        )
        _commit_mps()
        host_indices, host_values = [], []
        if self.host is not None:
            # Run the host half while the GPU half is in flight; nothing may
            # read a GPU result before this, or the two run back to back.
            # Coalescing costs no wall time here -- the host half finishes
            # well before the GPU one -- and shortens the serial tail.
            host = self.host(coords)
            if not host.is_coalesced():
                host = host.coalesce()
            host_indices.append(host._indices().to(torch.int32))
            host_values.append(host._values())
        indices = [layout.cpu() for layout in gpu_indices] + host_indices
        values = [layout.cpu() for layout in gpu_values] + host_values
        # Two GPU layouts would share keys; only a single one is distinct.
        gpu_nnz = sum(layout.shape[1] for layout in gpu_indices)
        self.unique_prefix_nnz = gpu_nnz if len(gpu_indices) <= 1 else None
        if not indices:
            return (
                torch.zeros((3, 0), dtype=torch.int32),
                torch.zeros(0, dtype=torch.float32),
            )
        return torch.cat(indices, dim=1), torch.cat(values)

    def forward_device_entries(self, coords):
        """Return the table as ``[(indices, values)]`` layouts on the GPU.

        For a packer that builds its interaction graph on the GPU. The GPU
        half stays where it was computed, one layout per dispatch; the host
        half is coalesced and copied up, so no (pose, rot1, rot2) key gets
        more than one entry from each half -- which keeps the graph builder's
        atomic sums independent of thread order.
        """
        gpu_indices, gpu_values, _, _ = self.gpu._weighted_entries_by_layout(
            coords.to(_MPS)
        )
        _commit_mps()
        entries = list(zip(gpu_indices, gpu_values))
        if self.host is not None:
            host = self.host(coords)
            if not host.is_coalesced():
                host = host.coalesce()
            entries.append(
                (
                    host._indices().to(torch.int32).to(_MPS),
                    host._values().to(_MPS),
                )
            )
        return entries

    def __call__(self, coords):
        indices, values = self.forward_sparse_entries(coords)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=self.shape,
            is_coalesced=False,
            check_invariants=False,
        )
