from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import threading
from typing import Callable, Dict, Sequence, TypeVar
import warnings

import torch

from tmol.types import Tensor

from tmol.database import ParameterDatabase
from tmol.database._yaml import safe_load
from tmol.score import ScoreType
from tmol.score.common import ZeroTermPoseScoringModule
from tmol.utility._device import resolve_device

# force registration of the terms with the ScoreTermFactory
from tmol.score.terms import *  # noqa: F401, F403
from tmol.score.terms import ScoreTermFactory
from tmol.pose import PoseStack

logger = logging.getLogger(__name__)

# Current .sfxn (score function weights YAML) format version.  Bump the major
# version on breaking schema changes; bump the minor version on
# backward-compatible additions.  The version string is written into every
# .sfxn file and checked on load.
SFXN_FORMAT_VERSION: str = "1.0"

# Exact CUDA tensor equality synchronizes with the host. Only pay that cost
# when one duplicate sparse index layout would retain at least 16 MiB. CPU
# equality has no synchronization penalty, so all matching layouts are tested.
_CUDA_ROTAMER_LAYOUT_DEDUP_MIN_BYTES = 16 * 1024 * 1024
_CPU_ROTAMER_SORTED_LAYOUT_MIN_NNZ = 4096
_MAX_CPU_SCORE_TERM_WORKERS = 4
_MAX_CPU_FUSED_SCORE_WORKERS = 16
_CPU_PARALLEL_SCORE_BACKWARD_MIN_COORD_ELEMENTS = 8192
# Independent CUDA terms overlap profitably for one large pose or a wide batch,
# but stream setup and coordination cost more than they save for small poses.
_CUDA_PARALLEL_SCORE_MIN_COORD_ELEMENTS = 20 * 1024
# Backward has more stream-coordination overhead than inference. Keep eager
# gradient scoring serial until the batch is large enough to amortize it.
_CUDA_PARALLEL_GRAD_SCORE_MIN_COORD_ELEMENTS = 100 * 1024
# Reducing LJ/LK and electrostatics to one weighted lane saves device work for
# wide CUDA workloads, but the extra native dispatch is slower for small eager
# calls. These cutoffs retain the established latency path for those calls.
_CUDA_WEIGHTED_FUSED_MIN_SINGLE_POSE_ATOMS = 15_000
_CUDA_WEIGHTED_FUSED_MIN_BATCH_ATOMS_PER_POSE = 2_000
_CUDA_WEIGHTED_FUSED_MIN_GRAD_COORD_ELEMENTS = 64 * 1024
_CPU_SCORE_TERM_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_CPU_SCORE_TERM_EXECUTOR_LOCK = threading.Lock()
_ScoreCallResult = TypeVar("_ScoreCallResult")


def _use_weighted_fused_score(
    coords: torch.Tensor,
    *,
    force_for_cuda_graph: bool = False,
    needs_gradient: bool | None = None,
) -> bool:
    """Select weighted native reduction without latency or stability regressions."""
    if force_for_cuda_graph:
        return True
    if coords.device.type == "cpu":
        if needs_gradient is None:
            needs_gradient = torch.is_grad_enabled() and coords.requires_grad
        # Apple's CPU backend needs independent gradient lanes to keep
        # heterogeneous minimization trajectories stable. Linux CPU backends
        # retain the faster compact weighted reduction.
        return not needs_gradient or sys.platform != "darwin"

    if needs_gradient is None:
        needs_gradient = torch.is_grad_enabled() and coords.requires_grad

    if (
        needs_gradient
        and coords.numel() >= _CUDA_WEIGHTED_FUSED_MIN_GRAD_COORD_ELEMENTS
    ):
        return True

    n_poses = coords.shape[0]
    atoms_per_pose = coords.shape[-2]
    if n_poses == 1:
        return atoms_per_pose >= _CUDA_WEIGHTED_FUSED_MIN_SINGLE_POSE_ATOMS
    return atoms_per_pose >= _CUDA_WEIGHTED_FUSED_MIN_BATCH_ATOMS_PER_POSE


def _rotamer_dispatch_cutoff_compatible(
    device_type: str,
    producer_cutoff: float,
    consumer_cutoff: float,
) -> bool:
    """Select exact layouts everywhere and larger-cutoff supersets on CUDA."""
    return producer_cutoff == consumer_cutoff or (
        device_type == "cuda" and producer_cutoff > consumer_cutoff
    )


def _cpu_score_term_executor(n_workers: int) -> ThreadPoolExecutor:
    """Return a process-local executor shared by rendered CPU scorers."""
    with _CPU_SCORE_TERM_EXECUTOR_LOCK:
        executor = _CPU_SCORE_TERM_EXECUTORS.get(n_workers)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="tmol-score"
            )
            _CPU_SCORE_TERM_EXECUTORS[n_workers] = executor
        return executor


def _cpu_score_term_worker_count(n_terms: int, device: torch.device) -> int:
    """Return the number of independent CPU score terms to run concurrently."""
    if device.type != "cpu":
        return 0
    n_threads = torch.get_num_threads()
    return min(_MAX_CPU_SCORE_TERM_WORKERS, n_threads, n_terms)


def _reset_cpu_score_term_executors_after_fork() -> None:
    """Discard parent-process thread pools in a forked child."""
    global _CPU_SCORE_TERM_EXECUTORS, _CPU_SCORE_TERM_EXECUTOR_LOCK
    _CPU_SCORE_TERM_EXECUTORS = {}
    _CPU_SCORE_TERM_EXECUTOR_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cpu_score_term_executors_after_fork)


def _score_call_in_thread(
    score_call: Callable[[torch.Tensor], _ScoreCallResult],
    coords: torch.Tensor,
    grad_enabled: bool,
    inference_mode_enabled: bool,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    autocast_cache_enabled: bool,
    shared_block_neighbors: torch.Tensor | None = None,
    fused_score_weights: torch.Tensor | None = None,
) -> _ScoreCallResult:
    """Evaluate one CPU score term with the caller's thread-local modes."""
    with (
        torch.inference_mode(inference_mode_enabled),
        torch.set_grad_enabled(grad_enabled),
        torch.autocast(
            "cpu",
            enabled=autocast_enabled,
            dtype=autocast_dtype,
            cache_enabled=autocast_cache_enabled,
        ),
    ):
        if fused_score_weights is not None:
            if shared_block_neighbors is None:
                return score_call(coords, fused_score_weights)
            return score_call(coords, shared_block_neighbors, fused_score_weights)
        if shared_block_neighbors is None:
            return score_call(coords)
        return score_call(coords, shared_block_neighbors)


def _score_grad_in_thread(
    scores: torch.Tensor,
    coords: torch.Tensor,
    grad_scores: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor | None:
    """Differentiate one independent CPU score-term graph."""
    if not scores.requires_grad:
        return None
    with torch.set_grad_enabled(create_graph):
        (term_grad,) = torch.autograd.grad(
            scores,
            coords,
            grad_scores,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
    return term_grad


class _ParallelScoreTerms(torch.autograd.Function):
    """Run CPU term forwards concurrently with deterministic accumulation."""

    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        shared_block_neighbors: torch.Tensor | None,
        executor: ThreadPoolExecutor,
        term_modules: Sequence[torch.nn.Module],
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
        autocast_cache_enabled: bool,
    ) -> torch.Tensor:
        term_coords = coords
        futures = [
            executor.submit(
                _score_call_in_thread,
                term,
                term_coords,
                True,
                False,
                autocast_enabled,
                autocast_dtype,
                autocast_cache_enabled,
                (
                    shared_block_neighbors
                    if getattr(term, "block_neighbor_cutoff", None) is not None
                    else None
                ),
            )
            for term in term_modules
        ]
        term_scores = tuple(future.result() for future in futures)
        ctx.term_coords = term_coords
        ctx.term_scores = term_scores
        ctx.term_sizes = tuple(scores.shape[0] for scores in term_scores)
        ctx.executor = executor
        ctx.parallel_backward = (
            coords.numel() >= _CPU_PARALLEL_SCORE_BACKWARD_MIN_COORD_ELEMENTS
        )
        return torch.cat(term_scores, dim=0).detach()

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        # Serial scoring creates the term autograd nodes in score-type order;
        # PyTorch visits those nodes in reverse order. Preserve that reduction
        # order so concurrent forward scheduling cannot perturb minimization.
        create_graph = torch.is_grad_enabled()
        term_grad_inputs = []
        offset = sum(ctx.term_sizes)
        for term_scores, term_size in zip(
            reversed(ctx.term_scores), reversed(ctx.term_sizes)
        ):
            offset -= term_size
            term_grad_inputs.append(
                (
                    term_scores,
                    ctx.term_coords,
                    grad_scores[offset : offset + term_size],
                    create_graph,
                )
            )

        if ctx.parallel_backward:
            term_grad_futures = [
                ctx.executor.submit(
                    _score_grad_in_thread,
                    *grad_input,
                )
                for grad_input in term_grad_inputs
            ]
            term_grads = [future.result() for future in term_grad_futures]
        else:
            term_grads = [
                _score_grad_in_thread(*grad_input) for grad_input in term_grad_inputs
            ]

        grad_coords = None
        for term_grad in term_grads:
            if term_grad is not None:
                grad_coords = (
                    term_grad if grad_coords is None else grad_coords + term_grad
                )
        return grad_coords, None, None, None, None, None, None


def _linearize_rotamer_indices(indices: torch.Tensor, n_rots: int) -> torch.Tensor:
    """Encode ``[pose, rotamer_i, rotamer_j]`` indices as sortable integers."""
    indices_64 = indices.to(torch.int64)
    return (indices_64[0] * n_rots + indices_64[1]) * n_rots + indices_64[2]


def _try_coalesce_cpu_rotamer_layouts(
    indices: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    n_poses: int,
    n_rots: int,
) -> torch.Tensor | None:
    """Merge sparse CPU layouts when the largest layout contains their union.

    Sorting one complete layout and mapping the smaller layouts into it avoids
    concatenating and sorting every repeated index. ``None`` requests the
    general sparse-coalesce path when that containment invariant does not hold.
    """
    if len(indices) < 2:
        return None

    largest = max(range(len(indices)), key=lambda i: indices[i].shape[1])
    if indices[largest].shape[1] < _CPU_ROTAMER_SORTED_LAYOUT_MIN_NNZ:
        return None

    keys = [_linearize_rotamer_indices(layout, n_rots) for layout in indices]
    sorted_keys, order = torch.sort(keys[largest])
    if sorted_keys.numel() > 1 and bool(torch.any(sorted_keys[1:] == sorted_keys[:-1])):
        return None

    combined_values = values[largest][order].clone()
    for layout_index, layout_keys in enumerate(keys):
        if layout_index == largest or layout_keys.numel() == 0:
            continue
        positions = torch.searchsorted(sorted_keys, layout_keys)
        if bool(torch.any(positions == sorted_keys.numel())) or not torch.equal(
            sorted_keys[positions], layout_keys
        ):
            return None
        combined_values.index_add_(0, positions, values[layout_index])

    return torch.sparse_coo_tensor(
        indices[largest][:, order],
        combined_values,
        size=(n_poses, n_rots, n_rots),
        is_coalesced=True,
        check_invariants=False,
    )


class ScoreFunction:
    """Weighted collection of energy terms rendered for a pose topology.

    Args:
        param_db: Chemical and scoring parameters used to construct terms.
        device: Device on which weights and rendered scorers operate. An
            unindexed CUDA device resolves to the current CUDA device.
    """

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        device = resolve_device(device)
        self._weights = torch.zeros((ScoreType.n_score_types.value,), device=device)

        self._all_terms = []
        self._all_terms_unordered = []
        self._all_terms_out_of_date = False

        self._all_score_types = []

        self._one_body_terms = []
        self._one_body_terms_unordered = []
        self._one_body_terms_out_of_date = False

        self._two_body_terms = []
        self._two_body_terms_unordered = []
        self._two_body_terms_out_of_date = False

        self._multi_body_terms = []
        self._multi_body_terms_unordered = []
        self._multi_body_terms_out_of_date = False

        self._weights_tensor_out_of_date = True
        self._weights_tensor = None
        self._weight_indices_tensor = None
        self._term_for_st = [None] * ScoreType.n_score_types.value
        self._param_db = param_db
        self._device = device
        self._terms_version = 0
        self._options_version = 0
        self._prepared_packed_block_types = None
        self._prepared_versions = None
        self._prepared_block_type_ids = None
        self._prepared_packed_annotation_names = ()
        self._setup_token = object()

        self.term_options = {}

    def set_weight(self, st: ScoreType, weight: float) -> None:
        """Set the weight for one score type.

        The energy term that implements ``st`` is created lazily when the
        requested weight is nonzero.

        Args:
            st: Score type to update.
            weight: New scalar weight.
        """
        # Do not construct an energy term merely to assign it a zero weight.
        # FastRelax updates its (usually disabled) constraint weight at every
        # schedule step; constructing that term adds a device-to-host sync to
        # every subsequent score evaluation even though it contributes zero.
        if weight == 0 and not self.score_type_covered_by_contained_term(st):
            self._weights[st.value] = weight
            self._weights_tensor_out_of_date = True
            return
        if not self.score_type_covered_by_contained_term(st):
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)
        self._weights[st.value] = weight
        self._weights_tensor_out_of_date = True

    def get_weight(self, st: ScoreType) -> torch.Tensor:
        """Return the scalar weight for ``st`` on the score-function device."""
        return self._weights[st.value]

    def score_type_covered_by_contained_term(self, st: ScoreType) -> bool:
        """Return whether a constructed energy term implements ``st``."""
        # `_all_terms` is a lazily refreshed sorted cache; consulting it while
        # weights are being populated can miss a term already present in the
        # unordered source lists and construct duplicate term objects.
        return self._term_for_st[st.value] is not None

    def retrieve_term_for_score_type(self, st: ScoreType):
        term = ScoreTermFactory.create_term_for_score_type(
            st, self._param_db, self._device
        )
        # sanity check: if the ScoreTermFactory returns the wrong term,
        # we want to know
        assert st in term.score_types()
        for tst in term.score_types():
            self._term_for_st[tst.value] = term
        self._all_terms_unordered.append(term)
        self._all_terms_out_of_date = True
        self._weight_indices_tensor = None
        self._terms_version += 1
        if term.n_bodies() == 1:
            self._one_body_terms_unordered.append(term)
            self._one_body_terms_out_of_date = True
        elif term.n_bodies() == 2:
            self._two_body_terms_unordered.append(term)
            self._two_body_terms_out_of_date = True
        else:
            self._multi_body_terms_unordered.append(term)
            self._multi_body_terms_out_of_date = True

    def term_for_st_has_no_other_non_zero_weights(self, st: ScoreType):
        term = self._term_for_st[st.value]
        for st2 in term.score_types():
            if st2 == st:
                continue
            if self._weights[st2.value] != 0:
                return False
        return True

    def remove_term_for_score_type(self, st: ScoreType):
        """Remove the term containing ``st`` once all its weights are zero."""
        term = self._term_for_st[st.value]
        if term is None:
            return

        self._all_terms_unordered.remove(term)
        self._all_terms_out_of_date = True
        self._weight_indices_tensor = None
        if term.n_bodies() == 1:
            self._one_body_terms_unordered.remove(term)
            self._one_body_terms_out_of_date = True
        elif term.n_bodies() == 2:
            self._two_body_terms_unordered.remove(term)
            self._two_body_terms_out_of_date = True
        else:
            self._multi_body_terms_unordered.remove(term)
            self._multi_body_terms_out_of_date = True

        for covered_st in term.score_types():
            self._term_for_st[covered_st.value] = None
        self._terms_version += 1

    def all_terms(self):
        """Grant read access to the list of terms.

        Do not modify this list directly
        """
        if self._all_terms_out_of_date:
            self._all_terms, self._all_score_types = self.get_sorted_terms(
                self._all_terms_unordered
            )
            self._all_terms_out_of_date = False

        return self._all_terms

    def all_score_types(self) -> list[ScoreType]:
        """Return score types in the same order as :meth:`all_terms`."""
        if self._all_terms_out_of_date:
            self._all_terms, self._all_score_types = self.get_sorted_terms(
                self._all_terms_unordered
            )
            self._all_terms_out_of_date = False

        return self._all_score_types

    def one_body_terms(self):
        """Return the active one-body energy terms in score-type order."""
        if self._one_body_terms_out_of_date:
            self._one_body_terms, _ = self.get_sorted_terms(
                self._one_body_terms_unordered
            )
            self._one_body_terms_out_of_date = False

        return self._one_body_terms

    def two_body_terms(self):
        """Return the active two-body energy terms in score-type order."""
        if self._two_body_terms_out_of_date:
            self._two_body_terms, _ = self.get_sorted_terms(
                self._two_body_terms_unordered
            )
            self._two_body_terms_out_of_date = False

        return self._two_body_terms

    def multi_body_terms(self):
        """Return the active multi-body energy terms in score-type order."""
        if self._multi_body_terms_out_of_date:
            self._multi_body_terms, _ = self.get_sorted_terms(
                self._multi_body_terms_unordered
            )
            self._multi_body_terms_out_of_date = False

        return self._multi_body_terms

    def render_whole_pose_scoring_module(
        self, pose_stack: PoseStack, cuda_graph: bool | str = False
    ) -> "WholePoseScoringModule":
        """Render a callable that repeatedly scores one fixed pose topology.

        The returned callable owns the term-specific ``torch.nn.Module`` objects
        for ``pose_stack``. Its default call returns weighted energies shaped
        ``[n_poses]`` as coordinates change during inference or minimization.

        Set ``cuda_graph`` to ``"forward"`` for repeated inference,
        ``"forward_backward"`` for repeated scoring with coordinate gradients,
        or ``True`` to capture both paths. Graph capture requires CUDA and a
        fixed coordinate shape, dtype, and device. Forward-only replay reuses
        its output buffer; clone an output that must survive the next call.
        """
        self.pre_work_initialization(pose_stack)
        term_modules = []
        for term in self.all_terms():
            module = term.render_whole_pose_scoring_module(pose_stack)
            # Retain the score-lane width without evaluating the module.  The
            # default weighted fused path uses this metadata to map its scalar
            # contribution back onto the live score-function weight buffer.
            module.n_score_types = len(term.score_types())
            term_modules.append(module)
        scoring_module = WholePoseScoringModule(self.weights_tensor(), term_modules)
        if cuda_graph:
            mode = "both" if cuda_graph is True else cuda_graph
            scoring_module.enable_cuda_graphs(pose_stack.coords, mode=mode)
        return scoring_module

    def render_block_pair_scoring_module(
        self, pose_stack: PoseStack, *, interaction_only: bool = False
    ) -> "BlockPairScoringModule":
        """Render a callable that retains scores for every residue-block pair.

        The default call returns weighted energies shaped
        ``[n_poses, max_n_blocks, max_n_blocks]``.

        Set ``interaction_only=True`` when only strictly off-diagonal block
        pairs will be consumed. Terms whose block-pair scores are known to be
        diagonal-only are then omitted. Diagonal entries in the returned
        matrix are incomplete in this mode.
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_block_pair_scoring_module(
                pose_stack, interaction_only=interaction_only
            )
            for t in self.all_terms()
        ]
        return BlockPairScoringModule(self.weights_tensor(), term_modules)

    def render_rotamer_scoring_module(
        self,
        pose_stack: PoseStack,
        rotamer_set: "RotamerSet",  # noqa: F405
    ) -> "RotamerScoringModule":
        """Render a weighted sparse scorer for one rotamer set.

        Args:
            pose_stack: Poses whose fixed background interacts with the
                rotamers.
            rotamer_set: Candidate conformers and their pose/block indexing.

        Returns:
            A callable that accepts rotamer coordinates and returns a sparse
            COO tensor shaped ``[n_poses, n_rotamers, n_rotamers]``. Call
            ``coalesce()`` before reading its indices or values; optimized CPU
            results may already be coalesced.
        """
        self.pre_work_initialization(pose_stack)
        term_modules = []
        for term in self.all_terms():
            module = term.render_rotamer_scoring_module(pose_stack, rotamer_set)
            module.n_score_types = len(term.score_types())
            term_modules.append(module)
        return RotamerScoringModule(self.weights_tensor(), term_modules)

    def pre_work_initialization(self, pose_stack: PoseStack) -> None:
        """Prepare active energy terms for a pose topology.

        Repeated calls reuse topology-dependent setup when neither the terms,
        their options, nor the packed block types have changed. A newly built
        PackedBlockTypes with the same ordered residue-type objects can reuse
        the previous packed annotations.

        Args:
            pose_stack: Poses whose topology will be scored.
        """
        # set_options must be first, since some of the logic that follows it
        # may depend on the options
        terms = self.all_terms()
        for energy_term in terms:
            energy_term.set_options(self.term_options)

        packed_block_types = pose_stack.packed_block_types
        versions = (self._terms_version, self._options_version)
        same_object = (
            packed_block_types is self._prepared_packed_block_types
            and versions == self._prepared_versions
            and getattr(packed_block_types, "_score_setup_token", None)
            is self._setup_token
        )
        if not same_object:
            block_type_ids = tuple(
                id(block_type) for block_type in packed_block_types.active_block_types
            )
            previous = self._prepared_packed_block_types
            same_block_types = (
                previous is not None
                and versions == self._prepared_versions
                and block_type_ids == self._prepared_block_type_ids
                and getattr(previous, "_score_setup_token", None) is self._setup_token
            )
            if same_block_types:
                for name in self._prepared_packed_annotation_names:
                    setattr(packed_block_types, name, getattr(previous, name))
            else:
                for block_type in packed_block_types.active_block_types:
                    for energy_term in terms:
                        energy_term.setup_block_type(block_type)

            attributes_before = dict(vars(packed_block_types))
            for energy_term in terms:
                energy_term.setup_packed_block_types(packed_block_types)
            annotations = {
                name
                for name, value in vars(packed_block_types).items()
                if name not in attributes_before or attributes_before[name] is not value
            }
            annotations.update(
                name
                for name in self._prepared_packed_annotation_names
                if hasattr(packed_block_types, name)
            )
            self._prepared_packed_annotation_names = tuple(sorted(annotations))
            self._prepared_block_type_ids = block_type_ids
            self._prepared_packed_block_types = packed_block_types
            self._prepared_versions = versions
            # Packed-block annotations can be score-function-specific (for
            # example, beta2016 and beta_soft reference energies). Record a
            # non-owning identity token so another score function using the
            # shared object invalidates this fast path.
            packed_block_types._score_setup_token = self._setup_token
        for energy_term in terms:
            energy_term.setup_poses(pose_stack)

    def set_option(self, key: str, value) -> None:
        """Set an option for all energy terms.

        Options are passed to each energy term's set_options method
        as a dictionary during pre_work_initialization.
        """
        self.term_options[key] = value
        self._options_version += 1

    def set_options(self, options: Dict) -> None:
        """Set the score function options by a dict.

        This replaces the options dict entirely - any previous values
        are gone.
        """
        self.term_options = options
        self._options_version += 1

    def weights_tensor(self) -> Tensor[torch.float32][:]:
        """Return weights aligned with the rendered term/subterm order."""
        if self._weights_tensor_out_of_date:
            # Keep weight collection on-device. Constructing a tensor from a
            # Python list of CUDA scalar tensors calls ``item()`` on every
            # entry, serializing the host with the scoring stream each time a
            # FastRelax stage changes a weight and renders a new scorer.
            if self._weight_indices_tensor is None:
                self._weight_indices_tensor = torch.tensor(
                    [
                        st.value
                        for term in self.all_terms()
                        for st in term.score_types()
                    ],
                    dtype=torch.int64,
                    device=self._device,
                )
            self._weights_tensor = self._weights[self._weight_indices_tensor]
            self._weights_tensor_out_of_date = False
        return self._weights_tensor

    @classmethod
    def from_sfxn_file(cls, path, param_db, device):
        """Create a ScoreFunction from a YAML weights file.

        Args:
            path: Path to a YAML file containing a ``weights`` dict mapping
                score type names (as in ``ScoreType``) to their weights, as well
                as any other options to configure the score function.
            param_db: ParameterDatabase instance.
            device: Target torch device.

        Returns:
            Configured ScoreFunction with all weights from the file applied.
        """
        with open(path) as f:
            data = safe_load(f)

        # --- .sfxn format version check ---
        file_version = data.get("version")
        if file_version is None:
            raise ValueError(
                f"{path}: no 'version' field found in .sfxn file. "
                f"Current format version is {SFXN_FORMAT_VERSION}. "
                f"Regenerate the file with the current version."
            )
        else:
            file_version = str(file_version)
            file_major = file_version.split(".")[0]
            current_major = SFXN_FORMAT_VERSION.split(".")[0]
            if file_major != current_major:
                raise ValueError(
                    f"{path}: .sfxn format version {file_version} is incompatible "
                    f"with the current format version {SFXN_FORMAT_VERSION}. "
                    f"Regenerate the file with the current writer."
                )
            if file_version != SFXN_FORMAT_VERSION:
                logger.info(
                    "%s: .sfxn format version %s differs from current %s "
                    "(backward-compatible minor version change)",
                    path,
                    file_version,
                    SFXN_FORMAT_VERSION,
                )

        sfxn = cls(param_db, device)
        for name, weight in data["weights"].items():
            sfxn.set_weight(getattr(ScoreType, name), weight)
        if "options" in data:
            sfxn.set_options(data["options"])
        return sfxn

    @staticmethod
    def get_sorted_terms(term_list):
        sorted_term_list = []
        sorted_score_type_list = []
        term_covered = [False] * ScoreType.n_score_types.value
        terms_by_st = [None] * ScoreType.n_score_types.value
        for term in term_list:
            for term_st in term.score_types():
                terms_by_st[term_st.value] = term

        for st_ind in range(ScoreType.n_score_types.value):
            if terms_by_st[st_ind] is not None:
                already_covered = False
                term = terms_by_st[st_ind]
                for term_st in term.score_types():
                    if term_covered[term_st.value]:
                        already_covered = True
                        break
                if not already_covered:
                    sorted_term_list.append(term)
                    for term_st in term.score_types():
                        term_covered[term_st.value] = True
                        sorted_score_type_list.append(term_st)
        return sorted_term_list, sorted_score_type_list


def _ljlk_elec_native_arguments(ljlk, elec, coords):
    """Collect dtype-adjusted arguments shared by LJ/LK--Elec fusion paths."""
    ljlk_tail = ljlk._static_tail_for_coords(coords)
    elec_tail = elec._static_tail_for_coords(coords)
    n_common = len(ljlk.common_parameters)
    common = ljlk_tail[:n_common]
    # Every ordinary term wrapper appends its block-pair-scoring flag.
    lp = ljlk_tail[n_common:-1]
    ep = elec_tail[len(elec.common_parameters) : -1]
    return (
        coords.flatten(start_dim=0, end_dim=-2),
        *common,
        lp[0],
        lp[1],
        lp[2],
        lp[5],
        lp[6],
        lp[7],
        lp[8],
        lp[9],
        lp[10],
        lp[11],
        ep[3],
        ep[6],
        ep[7],
        ep[9],
    )


class _FusedLJLKAndElecWholePoseModule(torch.nn.Module):
    """Internal execution group retaining four independent score lanes."""

    def __init__(self, ljlk_module, elec_module):
        super().__init__()
        self.ljlk_module = ljlk_module
        self.elec_module = elec_module
        self.classname = "LJLK+Elec"
        self.block_neighbor_cutoff = max(
            ljlk_module.block_neighbor_cutoff, elec_module.block_neighbor_cutoff
        )
        self.n_score_types = ljlk_module.n_score_types + elec_module.n_score_types

    def build_compact_block_neighbors(self, coords, reach):
        return self.ljlk_module.build_compact_block_neighbors(coords, reach)

    def _native_arguments(self, coords, shared_block_neighbors):
        return (
            *_ljlk_elec_native_arguments(self.ljlk_module, self.elec_module, coords),
            shared_block_neighbors,
        )

    def forward(self, coords, shared_block_neighbors=None):
        if shared_block_neighbors is None:
            return torch.cat(
                (self.ljlk_module(coords), self.elec_module(coords)), dim=0
            )
        from tmol.score.ljlk.potentials import ljlk_elec_pose_scores

        # Preserve the normal term wrapper's lazy float64 support.  The
        # composite selects arguments from each child's dtype-adjusted static
        # tail rather than bypassing it and reading raw float32 parameters.
        (scores,) = ljlk_elec_pose_scores(
            *self._native_arguments(coords, shared_block_neighbors)
        )
        return scores

    def forward_weighted(self, coords, shared_block_neighbors, score_weights):
        """Return one weighted lane while retaining the decomposed fallback."""
        from tmol.score.ljlk.potentials import ljlk_elec_weighted_pose_scores

        if score_weights.dtype != coords.dtype:
            score_weights = score_weights.to(dtype=coords.dtype)
        (scores,) = ljlk_elec_weighted_pose_scores(
            *self._native_arguments(coords, shared_block_neighbors), score_weights
        )
        return scores


def _whole_pose_execution_modules(term_modules, device):
    """Build conservative built-in execution groups without changing lanes."""
    grouped = []
    index = 0
    while index < len(term_modules):
        if index + 1 < len(term_modules):
            first = term_modules[index]
            second = term_modules[index + 1]
            is_builtin_pair = False
            if (
                getattr(first, "classname", None) == "LJLK"
                and getattr(second, "classname", None) == "Elec"
            ):
                from tmol.score.elec.potentials import elec_pose_scores
                from tmol.score.ljlk.potentials import ljlk_pose_scores

                is_builtin_pair = (
                    getattr(first, "term_score_poses", None) is ljlk_pose_scores
                    and getattr(second, "term_score_poses", None) is elec_pose_scores
                )
            if (
                is_builtin_pair
                and getattr(first, "block_neighbor_cutoff", None) is not None
                and getattr(second, "block_neighbor_cutoff", None) is not None
                and not any(
                    parameter.requires_grad
                    for term in (first, second)
                    for parameter in term.parameters()
                )
                and not (
                    device.type == "cpu"
                    and torch.get_num_threads() > 1
                    and getattr(first, "n_poses", None) != 1
                )
            ):
                grouped.append(_FusedLJLKAndElecWholePoseModule(first, second))
                index += 2
                continue
        grouped.append(term_modules[index])
        index += 1
    return tuple(grouped)


class WholePoseScoringModule:
    """Rendered energy modules that score complete poses."""

    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(weights.unsqueeze(1), requires_grad=False)
        self.term_modules = tuple(term_modules)
        self._execution_modules = _whole_pose_execution_modules(
            self.term_modules, weights.device
        )
        self._active_term_indices = tuple(
            index
            for index, term in enumerate(self._execution_modules)
            if not isinstance(term, ZeroTermPoseScoringModule)
        )
        self._has_trainable_term_parameters = any(
            parameter.requires_grad
            for term in self.term_modules
            for parameter in term.parameters()
        )
        self._cpu_term_workers = _cpu_score_term_worker_count(
            len(self._execution_modules), weights.device
        )
        self._fused_ljlk_elec_module_index = next(
            (
                index
                for index, term in enumerate(self._execution_modules)
                if isinstance(term, _FusedLJLKAndElecWholePoseModule)
            ),
            None,
        )
        self._fused_ljlk_elec_weight_range = None
        if self._fused_ljlk_elec_module_index is not None:
            fused_index = self._fused_ljlk_elec_module_index
            if all(
                hasattr(term, "n_score_types")
                for term in self._execution_modules[: fused_index + 1]
            ):
                weight_begin = sum(
                    term.n_score_types for term in self._execution_modules[:fused_index]
                )
                self._fused_ljlk_elec_weight_range = (
                    weight_begin,
                    weight_begin + self._execution_modules[fused_index].n_score_types,
                )
        cpu_threads = torch.get_num_threads() if weights.device.type == "cpu" else 1
        # Use the same compact execution policy for single- and multi-pose
        # scorers so batching cannot perturb nonlinear minimization trajectories.
        self._cpu_fused_shards = (
            min(8, max(2, cpu_threads // 2))
            if self._fused_ljlk_elec_module_index is not None and cpu_threads >= 2
            else 0
        )
        self._cpu_fused_workers = min(
            _MAX_CPU_FUSED_SCORE_WORKERS,
            cpu_threads,
            self._cpu_fused_shards + max(0, len(self._execution_modules) - 1),
        )
        self._cuda_term_streams: tuple[torch.cuda.Stream, ...] | None = None
        self._shared_neighbor_term_indices = tuple(
            index
            for index, term in enumerate(self.term_modules)
            if getattr(term, "block_neighbor_cutoff", None) is not None
        )
        self._shared_neighbor_cutoff = max(
            (
                self.term_modules[index].block_neighbor_cutoff
                for index in self._shared_neighbor_term_indices
            ),
            default=None,
        )

    def _build_shared_block_neighbors(
        self, coords: torch.Tensor
    ) -> torch.Tensor | None:
        if len(self._shared_neighbor_term_indices) < 2:
            return None
        builder = self.term_modules[self._shared_neighbor_term_indices[0]]
        return builder.build_compact_block_neighbors(
            coords, self._shared_neighbor_cutoff
        )

    @staticmethod
    def _call_term(term, coords, shared_block_neighbors, fused_score_weights=None):
        if fused_score_weights is not None:
            return term.forward_weighted(
                coords, shared_block_neighbors, fused_score_weights
            )
        if (
            shared_block_neighbors is not None
            and getattr(term, "block_neighbor_cutoff", None) is not None
        ):
            return term(coords, shared_block_neighbors)
        return term(coords)

    def _can_use_weighted_fused_default(self, coords: torch.Tensor) -> bool:
        """Whether default scoring can consume the fused scalar contribution."""
        needs_gradient = torch.is_grad_enabled() and coords.requires_grad
        # Weighted fusion intentionally discards the four independent native
        # lanes. Preserve the canonical Python weighting path when callers opt
        # into fitting score-function weights.
        if torch.is_grad_enabled() and self.weights.requires_grad:
            return False
        if self._fused_ljlk_elec_weight_range is None or not _use_weighted_fused_score(
            coords,
            force_for_cuda_graph=getattr(
                self, "_force_weighted_fusion_for_cuda_graph", False
            ),
            needs_gradient=needs_gradient,
        ):
            return False
        # The generic CPU parallel autograd wrapper expects canonical term-lane
        # widths. One-thread CPU, the explicit fused-shard path, and all CUDA
        # execution paths can consume the compact weighted lane directly.
        if coords.device.type == "cuda" or self._cpu_term_workers < 2:
            return True
        if self._cpu_fused_shards < 2:
            return False
        return needs_gradient or self._cpu_fused_shards <= 2

    def _reduce_weighted_fused_lanes(self, score_lanes: torch.Tensor) -> torch.Tensor:
        """Combine one already-weighted fused lane in one native reduction."""
        from tmol.score.ljlk.potentials import weighted_fused_score_sum

        weight_begin, weight_end = self._fused_ljlk_elec_weight_range
        score_weights = self.weights
        if score_weights.dtype != score_lanes.dtype:
            score_weights = score_weights.to(dtype=score_lanes.dtype)
        (scores,) = weighted_fused_score_sum(
            score_lanes, score_weights, weight_begin, weight_end - weight_begin
        )
        return scores

    def __call__(
        self,
        coords: torch.Tensor,
        sum_terms: bool = True,
        apply_weights: bool = True,
    ) -> torch.Tensor:
        if sum_terms and apply_weights:
            needs_grad = torch.is_grad_enabled() and coords.requires_grad
            if needs_grad and hasattr(self, "_cuda_graphed_autograd"):
                # Graphed callables reuse their output storage. Optimizers such
                # as LBFGS retain prior loss tensors across closure calls, so
                # return owned storage rather than allowing a later replay to
                # mutate an earlier loss.
                return self._cuda_graphed_autograd(coords).clone()
            if not needs_grad and hasattr(self, "_cuda_graphed_forward"):
                return self._cuda_graphed_forward(coords)
        if not torch.is_grad_enabled() and coords.requires_grad:
            coords = coords.detach()
        fused_score_weights = None
        if sum_terms and apply_weights and self._can_use_weighted_fused_default(coords):
            weight_begin, weight_end = self._fused_ljlk_elec_weight_range
            fused_score_weights = self.weights[weight_begin:weight_end, 0]
        unweighted = self.unweighted_scores(
            coords, fused_score_weights=fused_score_weights
        )
        if fused_score_weights is not None:
            return self._reduce_weighted_fused_lanes(unweighted)
        weighted = unweighted.mul_(self.weights) if apply_weights else unweighted
        summed = torch.sum(weighted, dim=0) if sum_terms else weighted

        return summed

    def unweighted_scores(
        self,
        coords: torch.Tensor,
        *,
        fused_score_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        needs_grad = torch.is_grad_enabled() and coords.requires_grad
        shared_block_neighbors = self._build_shared_block_neighbors(coords)
        execution_modules = (
            self._execution_modules
            if shared_block_neighbors is not None
            else self.term_modules
        )
        active_term_indices = tuple(
            index
            for index, term in enumerate(execution_modules)
            if not isinstance(term, ZeroTermPoseScoringModule)
        )
        parallel_min_elements = (
            _CUDA_PARALLEL_GRAD_SCORE_MIN_COORD_ELEMENTS
            if needs_grad
            else _CUDA_PARALLEL_SCORE_MIN_COORD_ELEMENTS
        )
        if (
            coords.device.type == "cuda"
            and not (torch.is_grad_enabled() and self._has_trainable_term_parameters)
            and len(active_term_indices) >= 2
            and coords.numel() >= parallel_min_elements
        ):
            cuda_scores = self._parallel_cuda_scores(
                coords,
                needs_grad,
                shared_block_neighbors,
                execution_modules,
                active_term_indices,
                fused_score_weights,
            )
            if cuda_scores is not None:
                return torch.cat(cuda_scores, dim=0)

        cpu_workers = self._cpu_term_workers
        if execution_modules is not self._execution_modules:
            cpu_workers = min(
                cpu_workers,
                _cpu_score_term_worker_count(len(execution_modules), coords.device),
            )
        if torch.is_grad_enabled() and self._has_trainable_term_parameters:
            cpu_workers = 0
        if (
            self._cpu_fused_shards >= 2
            and not self._has_trainable_term_parameters
            and shared_block_neighbors is not None
        ):
            fused_scores = self._parallel_cpu_fused_scores(
                coords,
                needs_grad,
                shared_block_neighbors,
                fused_score_weights,
            )
            if fused_scores is not None:
                return fused_scores
        if cpu_workers < 2:
            return torch.cat(
                [
                    self._call_term(
                        term,
                        coords,
                        shared_block_neighbors,
                        (
                            fused_score_weights
                            if index == self._fused_ljlk_elec_module_index
                            else None
                        ),
                    )
                    for index, term in enumerate(execution_modules)
                ],
                dim=0,
            )

        executor = _cpu_score_term_executor(cpu_workers)
        autocast_context = (
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        )
        if needs_grad:
            return _ParallelScoreTerms.apply(
                coords,
                shared_block_neighbors,
                executor,
                execution_modules,
                *autocast_context,
            )

        context = (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            *autocast_context,
        )
        futures = [
            executor.submit(
                _score_call_in_thread,
                term,
                coords,
                *context,
                (
                    shared_block_neighbors
                    if getattr(term, "block_neighbor_cutoff", None) is not None
                    else None
                ),
            )
            for term in execution_modules
        ]
        return torch.cat([future.result() for future in futures], dim=0)

    def _parallel_cpu_fused_scores(
        self,
        coords: torch.Tensor,
        needs_grad: bool,
        shared_block_neighbors: torch.Tensor,
        fused_score_weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Shard a fused pair traversal while other CPU terms run concurrently."""
        fused_index = self._fused_ljlk_elec_module_index
        if fused_index is None or self._cpu_fused_workers < 2:
            return None
        n_neighbors = int(shared_block_neighbors[0])
        desired_shards = self._cpu_fused_shards
        if coords.numel() < 16 * 1024:
            desired_shards = min(desired_shards, 4)
        n_shards = min(desired_shards, max(1, n_neighbors))
        if n_shards < 2 and fused_score_weights is None:
            return None

        neighbor_shards = []
        for shard in range(n_shards):
            begin = n_neighbors * shard // n_shards
            end = n_neighbors * (shard + 1) // n_shards
            neighbor_shards.append(
                torch.cat(
                    (
                        shared_block_neighbors.new_tensor([end - begin]),
                        shared_block_neighbors[1 + begin : 1 + end],
                    )
                )
            )

        executor = _cpu_score_term_executor(self._cpu_fused_workers)
        autocast_context = (
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        )
        context = (
            (True, False, *autocast_context)
            if needs_grad
            else (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
                *autocast_context,
            )
        )
        fused_term = self._execution_modules[fused_index]
        fused_score_call = (
            fused_term.forward_weighted
            if fused_score_weights is not None
            else fused_term
        )
        # Submit the expensive shards first. Remaining workers immediately
        # pick up independent terms, keeping both kinds of parallelism.
        fused_futures = [
            executor.submit(
                _score_call_in_thread,
                fused_score_call,
                coords,
                *context,
                shard,
                fused_score_weights,
            )
            for shard in neighbor_shards
        ]
        term_futures = {}
        for index, term in enumerate(self._execution_modules):
            if index == fused_index:
                continue
            term_futures[index] = executor.submit(
                _score_call_in_thread,
                term,
                coords,
                *context,
                (
                    shared_block_neighbors
                    if getattr(term, "block_neighbor_cutoff", None) is not None
                    else None
                ),
            )

        fused_scores = torch.stack(
            [future.result() for future in fused_futures], dim=0
        ).sum(dim=0)
        scores = []
        for index in range(len(self._execution_modules)):
            scores.append(
                fused_scores if index == fused_index else term_futures[index].result()
            )
        return torch.cat(scores, dim=0)

    def _parallel_cuda_scores(
        self,
        coords: torch.Tensor,
        needs_grad: bool,
        shared_block_neighbors: torch.Tensor | None,
        execution_modules: Sequence[torch.nn.Module],
        active_term_indices: Sequence[int],
        fused_score_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Run independent large-workload score terms on separate streams."""
        if self._cuda_term_streams is None or len(self._cuda_term_streams) != len(
            active_term_indices
        ):
            # Creating side streams during an unrelated user capture is unsafe.
            # Our graph wrapper warms this path first, so its streams already
            # exist when capture begins.
            with torch.cuda.device(coords.device):
                if torch.cuda.is_current_stream_capturing():
                    return None
            self._cuda_term_streams = tuple(
                torch.cuda.Stream(device=coords.device) for _ in active_term_indices
            )

        current = torch.cuda.current_stream(coords.device)
        scores: list[torch.Tensor | None] = [None] * len(execution_modules)
        # Distinct zero-copy views give every term a caller-stream autograd
        # node, synchronizing returned gradients before leaf accumulation.
        term_coords = (
            tuple(coords.view_as(coords) for _ in active_term_indices)
            if needs_grad
            else (coords,) * len(active_term_indices)
        )
        for index, term in enumerate(execution_modules):
            if isinstance(term, ZeroTermPoseScoringModule):
                scores[index] = term(coords)
        for stream, index, term_input in zip(
            self._cuda_term_streams, active_term_indices, term_coords
        ):
            stream.wait_stream(current)
            term_input.record_stream(stream)
            term = execution_modules[index]
            term_neighbors = (
                shared_block_neighbors
                if shared_block_neighbors is not None
                and getattr(term, "block_neighbor_cutoff", None) is not None
                else None
            )
            if term_neighbors is not None:
                term_neighbors.record_stream(stream)
            with torch.cuda.stream(stream):
                scores[index] = self._call_term(
                    term,
                    term_input,
                    term_neighbors,
                    (
                        fused_score_weights
                        if index == self._fused_ljlk_elec_module_index
                        else None
                    ),
                )

        for stream, index in zip(self._cuda_term_streams, active_term_indices):
            current.wait_stream(stream)
            score = scores[index]
            assert score is not None
            score.record_stream(current)

        complete_scores = []
        for score in scores:
            assert score is not None
            complete_scores.append(score)
        return tuple(complete_scores)

    def enable_cuda_graphs(
        self, example_coords: torch.Tensor, mode: str = "both"
    ) -> "WholePoseScoringModule":
        """Capture the default weighted score for a fixed coordinate shape.

        The returned scorer accepts new coordinate values with the same shape,
        dtype, and device and retains forward and backward support. Calls that
        request unweighted or unsummed terms continue to use the eager path.
        Forward-only replay reuses its output buffer.

        ``mode`` may be ``"forward"``, ``"forward_backward"``, or ``"both"``.
        Capture has a one-time cost and retains static buffers, so select only
        the paths that will be reused. Calling this method again is a no-op for
        paths that are already captured.
        """
        if not example_coords.is_cuda:
            raise ValueError("CUDA graphs require CUDA coordinates")
        if mode not in ("forward", "forward_backward", "both"):
            raise ValueError(f"unsupported CUDA graph mode: {mode!r}")

        if mode in ("forward", "both") and not hasattr(self, "_cuda_graphed_forward"):
            # Capture this scorer rather than the serial graph module so large
            # inference workloads retain independent term-stream overlap.
            fused = self._fused_ljlk_elec_module_index is not None
            self._cuda_forward_graph_uses_fused_execution = fused
            self._cuda_forward_graph_uses_weighted_fusion = fused
            self._force_weighted_fusion_for_cuda_graph = True
            try:
                self._cuda_graphed_forward = _InferenceCUDAGraph(self, example_coords)
            finally:
                del self._force_weighted_fusion_for_cuda_graph

        if mode in ("forward_backward", "both") and not hasattr(
            self, "_cuda_graphed_autograd"
        ):
            graph_module = _DefaultWholePoseScoringModule(
                self.weights,
                self._execution_modules,
                force_weighted_fusion_for_cuda_graph=True,
            )
            sample = example_coords.detach().clone().requires_grad_(True)
            with (
                torch.cuda.device(example_coords.device),
                torch.enable_grad(),
                warnings.catch_warnings(),
            ):
                # PyTorch's backward-capture warmup retains the sample leaf's
                # default-stream AccumulateGrad node. Capture and replay are
                # valid; suppress only that known internal warning.
                warnings.filterwarnings(
                    "ignore",
                    message="The AccumulateGrad node's stream does not match",
                )
                self._cuda_graphed_autograd = torch.cuda.make_graphed_callables(
                    graph_module, (sample,), allow_unused_input=True
                )
        return self


class _DefaultWholePoseScoringModule(torch.nn.Module):
    """Graph-capturable default reduction for a whole-pose scorer."""

    def __init__(
        self, weights, term_modules, *, force_weighted_fusion_for_cuda_graph=False
    ):
        super().__init__()
        self.weights = weights
        self.term_modules = torch.nn.ModuleList(term_modules)
        self._force_weighted_fusion_for_cuda_graph = (
            force_weighted_fusion_for_cuda_graph
        )
        self._shared_neighbor_terms = tuple(
            term
            for term in self.term_modules
            if getattr(term, "block_neighbor_cutoff", None) is not None
        )
        self._shared_neighbor_cutoff = max(
            (term.block_neighbor_cutoff for term in self._shared_neighbor_terms),
            default=None,
        )
        self._fused_module_index = next(
            (
                index
                for index, term in enumerate(self.term_modules)
                if isinstance(term, _FusedLJLKAndElecWholePoseModule)
            ),
            None,
        )
        self._fused_weight_range = None
        if self._fused_module_index is not None:
            weight_begin = sum(
                term.n_score_types
                for term in self.term_modules[: self._fused_module_index]
            )
            self._fused_weight_range = (
                weight_begin,
                weight_begin
                + self.term_modules[self._fused_module_index].n_score_types,
            )

    def forward(self, coords):
        shared_block_neighbors = None
        if len(self._shared_neighbor_terms) >= 2:
            builder = self._shared_neighbor_terms[0]
            shared_block_neighbors = builder.build_compact_block_neighbors(
                coords, self._shared_neighbor_cutoff
            )
        use_weighted_fusion = (
            self._fused_weight_range is not None
            and shared_block_neighbors is not None
            and _use_weighted_fused_score(
                coords,
                force_for_cuda_graph=self._force_weighted_fusion_for_cuda_graph,
            )
        )
        score_lanes = torch.cat(
            [
                WholePoseScoringModule._call_term(
                    term,
                    coords,
                    shared_block_neighbors,
                    (
                        self.weights[
                            self._fused_weight_range[0] : self._fused_weight_range[1],
                            0,
                        ]
                        if use_weighted_fusion and index == self._fused_module_index
                        else None
                    ),
                )
                for index, term in enumerate(self.term_modules)
            ],
            dim=0,
        )
        if not use_weighted_fusion:
            return torch.sum(self.weights * score_lanes, dim=0)

        from tmol.score.ljlk.potentials import weighted_fused_score_sum

        weight_begin, weight_end = self._fused_weight_range
        score_weights = self.weights
        if score_weights.dtype != score_lanes.dtype:
            score_weights = score_weights.to(dtype=score_lanes.dtype)
        (scores,) = weighted_fused_score_sum(
            score_lanes, score_weights, weight_begin, weight_end - weight_begin
        )
        return scores


class _InferenceCUDAGraph:
    """Forward-only graph replay with a fixed-address input buffer."""

    def __init__(self, module, example_coords):
        self._coords = example_coords.detach().clone()
        with torch.cuda.device(example_coords.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    module(self._coords)
            torch.cuda.current_stream().wait_stream(stream)

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=stream), torch.no_grad():
                self._output = module(self._coords)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape != self._coords.shape:
            raise ValueError(
                "CUDA graph coordinates must have shape "
                f"{tuple(self._coords.shape)}; got {tuple(coords.shape)}"
            )
        if coords.dtype != self._coords.dtype:
            raise TypeError(
                "CUDA graph coordinates must have dtype "
                f"{self._coords.dtype}; got {coords.dtype}"
            )
        if coords.device != self._coords.device:
            raise ValueError(
                "CUDA graph coordinates must be on "
                f"{self._coords.device}; got {coords.device}"
            )
        self._coords.copy_(coords)
        self._graph.replay()
        return self._output


class BlockPairScoringModule:
    """Rendered energy modules that retain per-block-pair scores."""

    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(
            weights.unsqueeze(1).unsqueeze(1).unsqueeze(1), requires_grad=False
        )
        self.term_modules = tuple(term_modules)
        self._active_term_modules = tuple(
            term
            for term in self.term_modules
            if not isinstance(term, ZeroTermPoseScoringModule)
        )
        self._has_trainable_term_parameters = any(
            parameter.requires_grad
            for term in self.term_modules
            for parameter in term.parameters()
        )
        self._cpu_term_workers = _cpu_score_term_worker_count(
            len(self.term_modules), weights.device
        )

    def _parallel_forward_scores(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, ...] | None:
        """Evaluate active CPU terms concurrently for a forward-only call."""
        cpu_workers = self._cpu_term_workers
        differentiable = torch.is_grad_enabled() and (
            coords.requires_grad or self._has_trainable_term_parameters
        )
        if cpu_workers < 2 or differentiable or len(self._active_term_modules) < 2:
            return None

        executor = _cpu_score_term_executor(cpu_workers)
        context = (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        )
        futures = [
            executor.submit(_score_call_in_thread, term, coords, *context)
            for term in self._active_term_modules
        ]
        return tuple(future.result() for future in futures)

    def __call__(
        self,
        coords: torch.Tensor,
        sum_terms: bool = True,
        apply_weights: bool = True,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled() and coords.requires_grad:
            coords = coords.detach()
        if sum_terms and apply_weights:
            parallel_scores = self._parallel_forward_scores(coords)
            active_results = (
                iter(parallel_scores) if parallel_scores is not None else None
            )

            active_scores = []
            active_weights = []
            weight_offset = 0
            for term in self.term_modules:
                if isinstance(term, ZeroTermPoseScoringModule):
                    weight_offset += term.shape[0]
                    continue
                scores = (
                    term(coords) if active_results is None else next(active_results)
                )
                next_offset = weight_offset + scores.shape[0]
                active_scores.append(scores)
                active_weights.append(self.weights[weight_offset:next_offset])
                weight_offset = next_offset
            if active_scores:
                unweighted = torch.cat(active_scores, dim=0)
                weights = torch.cat(active_weights, dim=0)
                return unweighted.mul_(weights).sum(dim=0)
        unweighted = self.unweighted_scores(coords)
        weighted = unweighted.mul_(self.weights) if apply_weights else unweighted
        summed = torch.sum(weighted, dim=0) if sum_terms else weighted

        return summed

    def score_interactions(
        self,
        coords: torch.Tensor,
        block_pair_indices: torch.Tensor,
        *,
        sum_terms: bool = True,
        apply_weights: bool = True,
    ) -> torch.Tensor:
        """Sum selected block-pair entries with one indexed reduction.

        Args:
            coords: Pose coordinates accepted by this rendered scorer.
            block_pair_indices: Shared block pairs shaped ``[n_pairs, 2]``.
                Each row contains ``(block_i, block_j)`` and is applied to
                every pose in the coordinate batch.
            sum_terms: Sum the score-type dimension when true.
            apply_weights: Apply the score function's weights when true.

        Returns:
            Scores shaped ``[n_poses]`` when ``sum_terms`` is true, otherwise
            ``[n_score_types, n_poses]``.
        """
        if block_pair_indices.ndim != 2 or block_pair_indices.shape[1] != 2:
            raise ValueError("block_pair_indices must have shape [n_pairs, 2]")
        if block_pair_indices.device != coords.device:
            raise ValueError(
                "block_pair_indices must be on the same device as coordinates"
            )
        if block_pair_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_pair_indices must have an integer dtype")
        block_i, block_j = block_pair_indices.to(torch.int64).unbind(dim=1)
        scores = self(coords, sum_terms=sum_terms, apply_weights=apply_weights)
        pair_dimension = 1 if sum_terms else 2
        return scores[..., block_i, block_j].sum(dim=pair_dimension)

    def unweighted_scores(self, coords: torch.Tensor) -> torch.Tensor:
        parallel_scores = self._parallel_forward_scores(coords)
        if parallel_scores is None:
            return torch.cat([term(coords) for term in self.term_modules], dim=0)

        active_results = iter(parallel_scores)
        return torch.cat(
            [
                (
                    term(coords)
                    if isinstance(term, ZeroTermPoseScoringModule)
                    else next(active_results)
                )
                for term in self.term_modules
            ],
            dim=0,
        )


class _FusedLJLKAndElecRotamerFunction(torch.autograd.Function):
    """Differentiate the compact fused table by one fused recomputation."""

    @staticmethod
    def forward(ctx, *args):
        from tmol.score.ljlk.potentials import ljlk_elec_weighted_rotamer_scores

        tensor_args = args[:-3]
        max_dis, score_weights, empty_dispatch = args[-3:]
        empty_gradients = score_weights.new_empty(0)
        scores, indices, _ = ljlk_elec_weighted_rotamer_scores(
            *tensor_args,
            max_dis,
            score_weights,
            empty_gradients,
            empty_dispatch,
        )
        saved_tensors = []
        ctx.scalar_args = {}
        for index, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                saved_tensors.append(arg)
            else:
                ctx.scalar_args[index] = arg
        ctx.save_for_backward(*saved_tensors, indices)
        ctx.n_inputs = len(args)
        ctx.mark_non_differentiable(indices)
        return scores, indices

    @staticmethod
    def backward(ctx, score_gradients, _):
        from tmol.score.ljlk.potentials import ljlk_elec_weighted_rotamer_scores

        saved_tensors = iter(ctx.saved_tensors[:-1])
        dispatch_indices = ctx.saved_tensors[-1]
        args = [
            ctx.scalar_args[index] if index in ctx.scalar_args else next(saved_tensors)
            for index in range(ctx.n_inputs)
        ]
        _, _, coord_gradients = ljlk_elec_weighted_rotamer_scores(
            *args[:-1], score_gradients.reshape(-1), dispatch_indices
        )
        return (coord_gradients,) + (None,) * (ctx.n_inputs - 1)


class _FusedLJLKAndElecRotamerModule(torch.nn.Module):
    """Internal group that emits one live-weighted sparse lane."""

    def __init__(self, ljlk_module, elec_module):
        super().__init__()
        self.ljlk_module = ljlk_module
        self.elec_module = elec_module
        self.classname = "LJLK+Elec"
        self.n_score_types = 4
        self.n_poses = ljlk_module.n_poses
        self.n_rots = ljlk_module.n_rots
        self.block_neighbor_cutoff = max(
            ljlk_module.block_neighbor_cutoff, elec_module.block_neighbor_cutoff
        )
        self.rotamer_dispatch_key = ljlk_module.rotamer_dispatch_key

    def is_compatible(self) -> bool:
        """Return whether the live modules still satisfy the fast-path contract."""
        return (
            self.ljlk_module.block_neighbor_cutoff == 6.0
            and self.elec_module.block_neighbor_cutoff == 5.5
            and not any(
                parameter.requires_grad
                for term in (self.ljlk_module, self.elec_module)
                for parameter in term.parameters()
            )
        )

    def _native_arguments(self, coords):
        ljlk = self.ljlk_module
        elec = self.elec_module
        return (
            *_ljlk_elec_native_arguments(ljlk, elec, coords),
            max(ljlk.block_neighbor_cutoff, elec.block_neighbor_cutoff),
        )

    def forward(self, coords, score_weights):
        from tmol.score.ljlk.potentials import ljlk_elec_weighted_rotamer_scores

        if score_weights.dtype != coords.dtype:
            score_weights = score_weights.to(dtype=coords.dtype)
        args = (
            *self._native_arguments(coords),
            score_weights,
            self.ljlk_module._empty_dispatch_indices,
        )
        if torch.is_grad_enabled() and coords.requires_grad:
            return _FusedLJLKAndElecRotamerFunction.apply(*args)
        scores, indices, _ = ljlk_elec_weighted_rotamer_scores(
            *args[:-1], score_weights.new_empty(0), args[-1]
        )
        return scores, indices


def _fused_ljlk_elec_rotamer_module(term_modules):
    """Return the canonical default LJ/LK+Elec packing group, if present."""
    weight_offset = 0
    for index, first in enumerate(term_modules[:-1]):
        second = term_modules[index + 1]
        first_width = getattr(first, "n_score_types", None)
        second_width = getattr(second, "n_score_types", None)
        if first_width is None or second_width is None:
            return None
        is_builtin_pair = False
        if (
            getattr(first, "classname", None) == "LJLK"
            and getattr(second, "classname", None) == "Elec"
        ):
            from tmol.score.elec.potentials import elec_rotamer_scores_shared
            from tmol.score.ljlk.potentials import ljlk_rotamer_scores

            is_builtin_pair = (
                getattr(first, "term_score_poses", None) is ljlk_rotamer_scores
                and getattr(second, "term_score_poses", None)
                is elec_rotamer_scores_shared
            )
        if (
            is_builtin_pair
            and first_width == 3
            and second_width == 1
            and first.block_neighbor_cutoff == 6.0
            and second.block_neighbor_cutoff == 5.5
            and not any(
                parameter.requires_grad
                for term in (first, second)
                for parameter in term.parameters()
            )
        ):
            return (
                _FusedLJLKAndElecRotamerModule(first, second),
                index,
                weight_offset,
            )
        weight_offset += first_width
    return None


class RotamerScoringModule:
    """Rendered energy modules that build sparse rotamer-pair energy tables.

    Large identical index layouts are combined before sparse coalescing to
    avoid retaining and sorting redundant block-pair indices.
    """

    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(
            weights.view(-1, 1, 1, 1), requires_grad=False
        )
        self.term_modules = tuple(term_modules)
        fused = _fused_ljlk_elec_rotamer_module(self.term_modules)
        self._fused_ljlk_elec = fused[0] if fused is not None else None
        self._fused_ljlk_elec_index = fused[1] if fused is not None else None
        self._fused_ljlk_elec_weight_offset = fused[2] if fused is not None else None
        self._has_trainable_term_parameters = any(
            parameter.requires_grad
            for term in self.term_modules
            for parameter in term.parameters()
        )
        self._cpu_term_workers = _cpu_score_term_worker_count(
            len(self.term_modules), weights.device
        )

    def _execution_terms(self, use_fused):
        """Return native calls, weights, and canonical-lane accounting."""
        execution_terms = []
        term_index = 0
        while term_index < len(self.term_modules):
            term = self.term_modules[term_index]
            if use_fused and term_index == self._fused_ljlk_elec_index:
                weight_begin = self._fused_ljlk_elec_weight_offset
                assert weight_begin is not None
                score_weights = self.weights[weight_begin : weight_begin + 4, 0, 0, 0]
                execution_terms.append((self._fused_ljlk_elec, score_weights, True))
                term_index += 2
            else:
                execution_terms.append((term, None, False))
                term_index += 1
        return execution_terms

    @staticmethod
    def _compatible_dispatch(term, dispatch_by_key, device_type):
        """Return the narrowest previously built compatible term layout.

        CUDA may reuse a larger-cutoff sphere-overlap layout because the
        consumer's native potential still applies its own distance cutoff.
        CPU reuse remains exact-cutoff only, as favored by measurements.
        """
        cutoff = getattr(term, "block_neighbor_cutoff", None)
        dispatch_key = getattr(term, "rotamer_dispatch_key", None)
        if (
            not getattr(term, "accepts_shared_dispatch", False)
            or dispatch_key is None
            or cutoff is None
        ):
            return None
        compatible = [
            (producer_cutoff, indices)
            for producer_cutoff, indices in dispatch_by_key.get(dispatch_key, ())
            if _rotamer_dispatch_cutoff_compatible(device_type, producer_cutoff, cutoff)
        ]
        return min(compatible, key=lambda item: item[0])[1] if compatible else None

    def _sequential_term_results(self, coords, execution_terms):
        """Evaluate terms in order while reusing compatible sparse dispatches."""
        dispatch_by_key = {}
        for term, score_weights, already_weighted in execution_terms:
            shared_dispatch = self._compatible_dispatch(
                term, dispatch_by_key, coords.device.type
            )
            if already_weighted:
                assert score_weights is not None
                result = term(coords, score_weights)
            elif shared_dispatch is not None:
                result = term.forward(coords, shared_dispatch)
            else:
                result = term.forward(coords)

            cutoff = getattr(term, "block_neighbor_cutoff", None)
            dispatch_key = getattr(term, "rotamer_dispatch_key", None)
            if dispatch_key is not None and cutoff is not None:
                dispatch_by_key.setdefault(dispatch_key, []).append((cutoff, result[1]))
            yield term, result, already_weighted
            # The consumer has retained the weighted values. Release the raw
            # lanes before the next native call allocates another score table.
            del result

    @staticmethod
    def _matching_layout(indices, all_indices, layouts_by_nnz):
        """Find an identical prior layout and whether content checks are enabled."""
        layout_index = next(
            (
                index
                for index, prior_indices in enumerate(all_indices)
                if indices.is_set_to(prior_indices)
            ),
            None,
        )
        compare_contents = (
            indices.device.type == "cpu"
            or indices.numel() * indices.element_size()
            >= _CUDA_ROTAMER_LAYOUT_DEDUP_MIN_BYTES
        )
        if layout_index is None and compare_contents:
            layout_index = next(
                (
                    index
                    for index in layouts_by_nnz.get(indices.shape[1], ())
                    if torch.equal(indices, all_indices[index])
                ),
                None,
            )
        return layout_index, compare_contents

    def _weighted_entries_by_layout(
        self, coords: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int | None, int | None]:
        """Evaluate terms and combine entries that have identical layouts.

        Keeping this representation outside a PyTorch sparse COO tensor is
        important to the packer: COO construction promotes coordinates to
        int64, and coalescing then needs another potentially very large sort.
        """
        if not torch.is_grad_enabled() and coords.requires_grad:
            coords = coords.detach()
        # Accumulate weighted values and their indices across all terms at the
        # dense [nnz] level.  This avoids torch.stack on sparse tensors, which
        # previously created a [n_subterms, n_poses, n_rots, n_rots] 4D sparse
        # tensor whose index storage grew as n_subterms × nnz × 4 int32.
        all_values: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        layouts_by_nnz: dict[int, list[int]] = {}
        n_poses: int | None = None
        n_rots: int | None = None
        weights_offset = 0

        differentiable = torch.is_grad_enabled() and (
            coords.requires_grad
            or self.weights.requires_grad
            or self._has_trainable_term_parameters
        )
        use_fused = (
            self._fused_ljlk_elec is not None
            and self._fused_ljlk_elec.is_compatible()
            and not self.weights.requires_grad
        )
        execution_terms = self._execution_terms(use_fused)
        parallel = self._cpu_term_workers >= 2 and not differentiable
        if parallel:
            executor = _cpu_score_term_executor(self._cpu_term_workers)
            context = (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cpu"),
                torch.get_autocast_dtype("cpu"),
                torch.is_autocast_cache_enabled(),
            )
            futures = [
                executor.submit(
                    _score_call_in_thread,
                    term.forward,
                    coords,
                    *context,
                    None,
                    score_weights,
                )
                for term, score_weights, _ in execution_terms
            ]
            term_results = (
                (term, future.result(), already_weighted)
                for (term, _, already_weighted), future in zip(execution_terms, futures)
            )
        else:
            # Do not retain every term's complete score/index tensors. CUDA
            # packing layouts can be many GiB apiece, so consume each result
            # before evaluating the next term.
            term_results = self._sequential_term_results(coords, execution_terms)

        for term, (scores, indices), already_weighted in term_results:
            # [n_subterms, nnz], [3, nnz]
            n_subterms = term.n_score_types if already_weighted else scores.shape[0]

            # Native rotamer terms already return compact int32 coordinates,
            # while sparse/Python terms (notably constraints) may inherit
            # PyTorch COO's int64 index dtype. Normalize only that uncommon
            # path and reject a custom term whose coordinates cannot be
            # represented by the native interaction graph.
            if indices.dtype != torch.int32:
                if indices.dtype != torch.int64:
                    raise TypeError(
                        "rotamer score indices must have dtype int32 or int64"
                    )
                if indices.numel() != 0:
                    min_index, max_index = torch.aminmax(indices)
                    if (
                        int(min_index) < 0
                        or int(max_index) > torch.iinfo(torch.int32).max
                    ):
                        raise OverflowError(
                            "rotamer score indices exceed the native int32 range"
                        )
                indices = indices.to(torch.int32)

            # Apply per-subterm weights and sum to [nnz] — no sparse tensor yet.
            if already_weighted:
                weighted_values = scores[0]
            else:
                w = self.weights[weights_offset : weights_offset + n_subterms, 0, 0, 0]
                weighted_values = (w[:, None] * scores).sum(dim=0)

            # Several terms share the same block-pair dispatch. Pointer
            # identity is free to check at every size; reserve the device-wide
            # equality comparison for layouts large enough to recover its
            # synchronization cost.
            layout_index, compare_layout_contents = self._matching_layout(
                indices, all_indices, layouts_by_nnz
            )
            if layout_index is None:
                layout_index = len(all_indices)
                all_values.append(weighted_values)
                all_indices.append(indices)
                if compare_layout_contents:
                    layouts_by_nnz.setdefault(indices.shape[1], []).append(layout_index)
            else:
                all_values[layout_index] = all_values[layout_index] + weighted_values
            weights_offset += n_subterms

            if n_poses is None:
                n_poses = term.n_poses
                n_rots = term.n_rots
            # Loop locals otherwise keep the previous raw table (and, after
            # merging layouts, its temporary weighted values) alive during
            # the next call to the result generator. Autograd retains any
            # tensors it still needs for differentiable scoring.
            del scores, weighted_values

        return all_indices, all_values, n_poses, n_rots

    def forward_sparse_entries(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted, uncoalesced ``(indices, values)`` for packing.

        Duplicate coordinates may remain when two score terms use different
        layouts. Consumers must accumulate rather than assign their values.
        Indices remain int32 so the packer avoids an unnecessary int64 COO
        round-trip.
        """
        all_indices, all_values, _, _ = self._weighted_entries_by_layout(coords)
        if not all_indices:
            return (
                torch.zeros((3, 0), dtype=torch.int32, device=coords.device),
                torch.zeros(0, dtype=torch.float32, device=coords.device),
            )
        return torch.cat(all_indices, dim=1), torch.cat(all_values)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        all_indices, all_values, n_poses, n_rots = self._weighted_entries_by_layout(
            coords
        )

        if n_poses is None:
            # No terms at all
            return torch.sparse_coo_tensor(
                torch.zeros((3, 0), dtype=torch.int32, device=coords.device),
                torch.zeros(0, dtype=torch.float32, device=coords.device),
                size=(0, 0, 0),
                is_coalesced=True,
                check_invariants=False,
            )
        assert n_rots is not None

        if coords.device.type == "cpu":
            coalesced = _try_coalesce_cpu_rotamer_layouts(
                all_indices, all_values, n_poses, n_rots
            )
            if coalesced is not None:
                return coalesced

        combined_values = torch.cat(all_values)
        combined_indices = torch.cat(all_indices, dim=1)
        return torch.sparse_coo_tensor(
            combined_indices,
            combined_values,
            size=(n_poses, n_rots, n_rots),
            is_coalesced=False,
            check_invariants=False,
        )
