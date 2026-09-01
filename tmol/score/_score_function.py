from concurrent.futures import ThreadPoolExecutor
import logging
import os
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
_CPU_PARALLEL_SCORE_BACKWARD_MIN_COORD_ELEMENTS = 8192
_CPU_SCORE_TERM_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_CPU_SCORE_TERM_EXECUTOR_LOCK = threading.Lock()
_ScoreCallResult = TypeVar("_ScoreCallResult")


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
        return score_call(coords)


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
        return grad_coords, None, None, None, None, None


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

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack, cuda_graph=False):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module. This
        object's __call__ will return a tensor of weighted energies of
        shape (n_poses,).

        Set ``cuda_graph`` to ``"forward"`` for repeated inference,
        ``"forward_backward"`` for repeated scoring with coordinate gradients,
        or ``True`` to capture both paths. Graph capture requires CUDA and a
        fixed coordinate shape, dtype, and device. Forward-only replay reuses
        its output buffer; clone an output that must survive the next call.
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_whole_pose_scoring_module(pose_stack) for t in self.all_terms()
        ]
        scoring_module = WholePoseScoringModule(self.weights_tensor(), term_modules)
        if cuda_graph:
            mode = "both" if cuda_graph is True else cuda_graph
            scoring_module.enable_cuda_graphs(pose_stack.coords, mode=mode)
        return scoring_module

    def render_block_pair_scoring_module(self, pose_stack: PoseStack):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module. This
        object's __call__ will return a tensor of weighted energies of
        shape (n_poses, max_n_blocks, max_n_blocks).
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_block_pair_scoring_module(pose_stack) for t in self.all_terms()
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
            A callable that accepts rotamer coordinates and returns an
            uncoalesced sparse COO tensor shaped
            ``[n_poses, n_rotamers, n_rotamers]``. Call ``coalesce()`` before
            reading its indices or values.
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_rotamer_scoring_module(pose_stack, rotamer_set)
            for t in self.all_terms()
        ]
        return RotamerScoringModule(self.weights_tensor(), term_modules)

    def pre_work_initialization(self, pose_stack: PoseStack) -> None:
        """Prepare active energy terms for a pose topology.

        Repeated calls reuse topology-dependent setup when neither the terms,
        their options, nor the packed block types have changed.

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
        if not (
            packed_block_types is self._prepared_packed_block_types
            and versions == self._prepared_versions
            and getattr(packed_block_types, "_score_setup_token", None)
            is self._setup_token
        ):
            for block_type in packed_block_types.active_block_types:
                for energy_term in terms:
                    energy_term.setup_block_type(block_type)
            for energy_term in terms:
                energy_term.setup_packed_block_types(packed_block_types)
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


class WholePoseScoringModule:
    """Rendered energy modules that score complete poses."""

    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(weights.unsqueeze(1), requires_grad=False)
        self.term_modules = tuple(term_modules)
        cpu_workers = min(
            _MAX_CPU_SCORE_TERM_WORKERS,
            torch.get_num_threads(),
            len(self.term_modules),
        )
        self._cpu_term_workers = cpu_workers if weights.device.type == "cpu" else 0

    def __call__(self, coords, sum_terms=True, apply_weights=True):
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
        unweighted = self.unweighted_scores(coords)
        weighted = unweighted.mul_(self.weights) if apply_weights else unweighted
        summed = torch.sum(weighted, dim=0) if sum_terms else weighted

        return summed

    def unweighted_scores(self, coords):
        if self._cpu_term_workers < 2:
            return torch.cat([term(coords) for term in self.term_modules], dim=0)

        executor = _cpu_score_term_executor(self._cpu_term_workers)
        autocast_context = (
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        )
        if torch.is_grad_enabled() and coords.requires_grad:
            return _ParallelScoreTerms.apply(
                coords,
                executor,
                self.term_modules,
                *autocast_context,
            )

        context = (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            *autocast_context,
        )
        futures = [
            executor.submit(_score_call_in_thread, term, coords, *context)
            for term in self.term_modules
        ]
        return torch.cat([future.result() for future in futures], dim=0)

    def enable_cuda_graphs(self, example_coords, mode="both"):
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
            graph_module = _DefaultWholePoseScoringModule(
                self.weights, self.term_modules
            )
            self._cuda_graphed_forward = _InferenceCUDAGraph(
                graph_module, example_coords
            )

        if mode in ("forward_backward", "both") and not hasattr(
            self, "_cuda_graphed_autograd"
        ):
            graph_module = _DefaultWholePoseScoringModule(
                self.weights, self.term_modules
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

    def __init__(self, weights, term_modules):
        super().__init__()
        self.weights = weights
        self.term_modules = torch.nn.ModuleList(term_modules)

    def forward(self, coords):
        unweighted = torch.cat([term(coords) for term in self.term_modules], dim=0)
        return torch.sum(self.weights * unweighted, dim=0)


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

    def __call__(self, coords):
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
        self.term_modules = term_modules

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        if not torch.is_grad_enabled() and coords.requires_grad:
            coords = coords.detach()
        if sum_terms and apply_weights:
            active_scores = []
            active_weights = []
            weight_offset = 0
            for term in self.term_modules:
                if isinstance(term, ZeroTermPoseScoringModule):
                    weight_offset += term.shape[0]
                    continue
                scores = term(coords)
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

    def unweighted_scores(self, coords):
        return torch.cat([term(coords) for term in self.term_modules], dim=0)


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
        cpu_workers = min(
            _MAX_CPU_SCORE_TERM_WORKERS,
            torch.get_num_threads(),
            len(self.term_modules),
        )
        self._cpu_term_workers = cpu_workers if weights.device.type == "cpu" else 0

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
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

        parallel = self._cpu_term_workers >= 2 and not (
            torch.is_grad_enabled() and coords.requires_grad
        )
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
                )
                for term in self.term_modules
            ]
            term_results = [future.result() for future in futures]
        else:
            term_results = [term.forward(coords) for term in self.term_modules]

        for term, (scores, indices) in zip(self.term_modules, term_results):
            # [n_subterms, nnz], [3, nnz]
            n_subterms = scores.shape[0]

            # Apply per-subterm weights and sum to [nnz] — no sparse tensor yet.
            w = self.weights[weights_offset : weights_offset + n_subterms, 0, 0, 0]
            weighted_values = (w[:, None] * scores).sum(dim=0)

            # Several terms share the same block-pair dispatch. Combine their
            # values now so the sparse coalesce does not sort another copy of
            # the same, potentially multi-gigabyte, index layout.
            deduplicate_layout = (
                indices.device.type == "cpu"
                or indices.numel() * indices.element_size()
                >= _CUDA_ROTAMER_LAYOUT_DEDUP_MIN_BYTES
            )
            candidate_layouts = (
                layouts_by_nnz.get(indices.shape[1], ()) if deduplicate_layout else ()
            )
            for layout_index in candidate_layouts:
                if torch.equal(indices, all_indices[layout_index]):
                    all_values[layout_index] = (
                        all_values[layout_index] + weighted_values
                    )
                    break
            else:
                layout_index = len(all_indices)
                all_values.append(weighted_values)
                all_indices.append(indices)
                if deduplicate_layout:
                    layouts_by_nnz.setdefault(indices.shape[1], []).append(layout_index)
            weights_offset += n_subterms

            if n_poses is None:
                n_poses = term.n_poses
                n_rots = term.n_rots

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
