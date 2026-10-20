import warnings
from copy import deepcopy

import attrs
import numpy
import torch

from tmol.pose import PoseStack
from tmol.score import ScoreFunction
from tmol.kinematics import (
    FoldForest,
    MoveMap,
    NodeType,
    BondDOFTypes,
    JumpDOFTypes,
    PoseStackKinematicsModule,
)
from tmol.kinematics.compiled import inverse_kin


class CartesianSfxnNetwork(torch.nn.Module):
    """Differentiable score network over selected Cartesian coordinates."""

    def __init__(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        coord_mask=None,
        cuda_graph: bool | str = False,
    ):
        super(CartesianSfxnNetwork, self).__init__()

        wpsm = score_function.render_whole_pose_scoring_module(
            pose_stack, cuda_graph=cuda_graph
        )
        self.whole_pose_scoring_module = wpsm
        self._score_function = score_function
        self._score_function_versions = (
            score_function._terms_version,
            score_function._options_version,
        )

        self.pose_stack = pose_stack
        # clone: forward() writes into full_coords in place, which would
        # otherwise overwrite the caller's coordinates
        self.full_coords = pose_stack.coords.clone().detach()
        self._uses_default_coord_mask = coord_mask is None
        if coord_mask is None:
            # Padding coordinates never contribute to a pose's score. Excluding
            # them keeps heterogeneous batches from allocating and updating
            # meaningless optimizer degrees of freedom.
            coord_mask = pose_stack.real_atoms
        self.coord_mask = coord_mask

        # Precompute flat integer indices for the boolean mask
        # Flat integer is faster than bool mask
        #   (since torch does nonzero each time under the hood)
        self._coord_flat_idx = (
            self.coord_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )
        self._all_coords_movable = (
            self._coord_flat_idx.numel() == self.coord_mask.numel()
        )

        self.masked_coords = torch.nn.Parameter(
            self.full_coords.view(-1, self.full_coords.shape[-1])[self._coord_flat_idx]
        )
        if self._all_coords_movable:
            self.full_coords = self.masked_coords.view_as(self.full_coords)

        # pose each element of masked_coords belongs to, for per-pose minimization
        pose_for_atom = torch.div(
            self._coord_flat_idx,
            self.full_coords.shape[1],
            rounding_mode="floor",
        )
        self.segment_ids = pose_for_atom.repeat_interleave(self.full_coords.shape[-1])

    def _reusable_for(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        coord_mask=None,
    ) -> bool:
        """Return whether this network can score another pose without rendering."""
        # Check cheap Python identity and shape predicates before any tensor
        # comparison. The normal repeated-topology path shares these tensors,
        # so it avoids both rebuilding ``real_atoms`` and synchronizing CUDA.
        if not (
            score_function is self._score_function
            and self._score_function_versions
            == (
                self._score_function._terms_version,
                self._score_function._options_version,
            )
            and pose_stack.packed_block_types is self.pose_stack.packed_block_types
            and pose_stack.constraint_set is self.pose_stack.constraint_set
            and pose_stack.coords.shape == self.pose_stack.coords.shape
            and pose_stack.inter_residue_connections
            is self.pose_stack.inter_residue_connections
            and pose_stack.inter_block_bondsep is self.pose_stack.inter_block_bondsep
        ):
            return False

        if coord_mask is None:
            if not self._uses_default_coord_mask:
                return False
        elif coord_mask is not self.coord_mask and not torch.equal(
            coord_mask, self.coord_mask
        ):
            return False

        return (
            pose_stack.block_type_ind is self.pose_stack.block_type_ind
            or torch.equal(pose_stack.block_type_ind, self.pose_stack.block_type_ind)
        ) and (
            pose_stack.block_coord_offset is self.pose_stack.block_coord_offset
            or torch.equal(
                pose_stack.block_coord_offset, self.pose_stack.block_coord_offset
            )
        )

    def _reset(
        self, score_function: ScoreFunction, pose_stack: PoseStack, coord_mask=None
    ) -> bool:
        """Reset for a compatible pose topology; report whether it was reused."""
        if not self._reusable_for(score_function, pose_stack, coord_mask):
            return False

        self.pose_stack = pose_stack
        with torch.no_grad():
            self.whole_pose_scoring_module.weights.copy_(
                score_function.weights_tensor().unsqueeze(1)
            )
            if self._all_coords_movable:
                self.masked_coords.copy_(pose_stack.coords.reshape(-1, 3))
            else:
                self.full_coords.copy_(pose_stack.coords)
                self.masked_coords.copy_(
                    pose_stack.coords.reshape(-1, 3)[self._coord_flat_idx]
                )
        self.masked_coords.grad = None
        return True

    def forward(self) -> torch.Tensor:
        if not self._all_coords_movable:
            self.full_coords = self.full_coords.detach()
            self.full_coords.view(-1, self.full_coords.shape[-1])[
                self._coord_flat_idx
            ] = self.masked_coords
        return self.whole_pose_scoring_module(self.full_coords)

    def pose_stack_from_dofs(self) -> PoseStack:
        if self._all_coords_movable:
            full_coords = self.masked_coords.detach().view_as(self.full_coords).clone()
            return attrs.evolve(self.pose_stack, coords=full_coords)
        full_coords = self.full_coords.detach().clone()
        full_coords.view(-1, full_coords.shape[-1])[
            self._coord_flat_idx
        ] = self.masked_coords.detach()
        return attrs.evolve(self.pose_stack, coords=full_coords)


def _same_fold_forest(left: FoldForest, right: FoldForest) -> bool:
    return (
        left.max_n_edges == right.max_n_edges
        and numpy.array_equal(left.n_edges, right.n_edges)
        and numpy.array_equal(left.edges, right.edges)
    )


def _same_move_map(left: MoveMap, right: MoveMap) -> bool:
    for field in attrs.fields(MoveMap):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, torch.Tensor):
            if not (
                left_value.shape == right_value.shape
                and left_value.device == right_value.device
                and torch.equal(left_value, right_value)
            ):
                return False
        elif left_value != right_value:
            return False
    return True


class _KinematicScoringModule(torch.nn.Module):
    """Map movable kinematic DOFs through coordinates to weighted pose scores.

    Every step is out of place so the same module can run eagerly or be
    captured as a CUDA graph. A graph reads ``base_dofs`` and ``base_coords``
    by address on replay, so update them in place rather than reassigning.
    """

    def __init__(
        self,
        kin_module: PoseStackKinematicsModule,
        scoring_module,
        base_dofs: torch.Tensor,
        base_coords: torch.Tensor,
        dof_flat_idx: torch.Tensor,
        atom_idx: torch.Tensor,
    ):
        super().__init__()
        self.kin_module = kin_module
        self.scoring_module = scoring_module
        self.base_dofs = base_dofs
        self.base_coords = base_coords
        self.dof_flat_idx = dof_flat_idx
        self.atom_idx = atom_idx

    def dofs(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        return (
            self.base_dofs.view(-1)
            .index_put((self.dof_flat_idx,), masked_dofs)
            .view_as(self.base_dofs)
        )

    def coords(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        # Node 0 is the kinematic root; it has no pose-stack atom.
        kin_coords = self.kin_module(self.dofs(masked_dofs))
        flat_coords = self.base_coords.view(-1, 3).index_put(
            (self.atom_idx,), kin_coords[1:].to(self.base_coords.dtype)
        )
        return flat_coords.view_as(self.base_coords)

    def forward(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        return self.scoring_module(self.coords(masked_dofs))


class KinForestSfxnNetwork(torch.nn.Module):
    """Differentiable score network over selected kinematic degrees of freedom.

    Set ``cuda_graph`` to ``"forward_backward"`` (or ``True``) to capture the
    complete DOF-to-score evaluation and its gradient as CUDA graphs,
    ``"forward"`` to capture gradient-free evaluation, or ``"both"``. Capture
    requires CUDA and a fixed topology; :meth:`_reset` reuses the captured
    graphs for new coordinates. Pass ``fold_forest`` and ``move_map`` to
    allow that reuse.
    """

    def __init__(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        kin_module: PoseStackKinematicsModule,
        dof_mask=None,
        kin_dtype=torch.float32,
        *,
        cuda_graph: bool | str = False,
        fold_forest: FoldForest | None = None,
        move_map: MoveMap | None = None,
    ):
        super(KinForestSfxnNetwork, self).__init__()

        torch_device = pose_stack.device
        self.pose_stack = pose_stack
        # The kinematic graph captures scoring itself; a separate
        # coordinate-input scoring graph would only retain more memory.
        wpsm = score_function.render_whole_pose_scoring_module(pose_stack)
        kmd = kin_module.kmd
        self.ff_copy = None if fold_forest is None else deepcopy(fold_forest)
        self.mm_copy = None if move_map is None else deepcopy(move_map)
        self.kin_module = kin_module
        self.whole_pose_scoring_module = wpsm
        self.kin_dtype = kin_dtype

        self._score_function = score_function
        self._score_function_versions = (
            score_function._terms_version,
            score_function._options_version,
        )

        self.orig_coords_shape = pose_stack.coords.shape
        self.id = kmd.forest.id
        self._atom_idx = self.id[1:].to(torch.int64)

        base_coords = pose_stack.coords.clone().detach()
        raw_dofs = self._dofs_from_coords(base_coords)

        if dof_mask is None:
            # Default behavior:
            #   Enable minimization of phi_c dofs for bonded atoms
            #   Enable minimization of 6 dofs for jump atoms
            #   - RBx, y, z, and
            #   - RBdel_alpha, beta, gamma
            dof_mask = torch.zeros(
                raw_dofs.shape, dtype=torch.bool, device=torch_device
            )
            dof_mask[kmd.forest.doftype == NodeType.bond, BondDOFTypes.phi_c] = True
            dof_mask[
                kmd.forest.doftype == NodeType.jump, : JumpDOFTypes.RBdel_gamma
            ] = True
        self.dof_mask = dof_mask

        # Precompute flat integer indices for the boolean mask
        # Flat integer is faster than bool mask
        self._dof_flat_idx = (
            self.dof_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )

        self.masked_dofs = torch.nn.Parameter(raw_dofs[self.dof_mask])

        # pose each element of masked_dofs belongs to, for per-pose minimization.
        # kmd.forest.id is the index into the flattened (pose, atom) coords; the
        # root node's id of -1 never survives the dof mask.
        n_atoms_per_pose = pose_stack.coords.shape[1]
        pose_for_node = torch.div(self.id, n_atoms_per_pose, rounding_mode="floor")
        pose_for_dof = pose_for_node.unsqueeze(1).expand(raw_dofs.shape).reshape(-1)
        self.segment_ids = pose_for_dof[self._dof_flat_idx].to(torch.int64)
        assert bool((self.segment_ids >= 0).all()), "root node dofs cannot be minimized"

        self._kinematic_scoring = _KinematicScoringModule(
            kin_module,
            wpsm,
            raw_dofs,
            base_coords,
            self._dof_flat_idx,
            self._atom_idx,
        )
        # A plain dict keeps the captured modules, and the score weights they
        # hold, out of parameters(): optimizers see only masked_dofs.
        self._cuda_graphs: dict[str, object] = {}
        if cuda_graph:
            mode = "forward_backward" if cuda_graph is True else cuda_graph
            self.enable_cuda_graphs(mode)

    def _dofs_from_coords(self, coords: torch.Tensor) -> torch.Tensor:
        kmd = self.kin_module.kmd
        kincoords = torch.zeros(
            (kmd.forest.id.shape[0], 3), dtype=self.kin_dtype, device=coords.device
        )
        kincoords[1:] = coords.view(-1, 3)[self._atom_idx].to(self.kin_dtype)
        return inverse_kin(
            kincoords,
            kmd.forest.parent,
            kmd.forest.frame_x,
            kmd.forest.frame_y,
            kmd.forest.frame_z,
            kmd.forest.doftype,
        )

    def enable_cuda_graphs(
        self, mode: str = "forward_backward"
    ) -> "KinForestSfxnNetwork":
        """Capture DOF-to-score evaluation for this fixed topology.

        Captures forward kinematics, the coordinate scatter, and weighted
        scoring as one graph and, for gradients, their backward pass as a
        second graph sharing its memory pool. ``mode`` may be ``"forward"``,
        ``"forward_backward"``, or ``"both"``; paths already captured are kept.
        """
        if not self.masked_dofs.is_cuda:
            raise ValueError("CUDA graphs require CUDA degrees of freedom")
        if mode not in ("forward", "forward_backward", "both"):
            raise ValueError(f"unsupported CUDA graph mode: {mode!r}")

        from tmol.score._score_function import _InferenceCUDAGraph
        from tmol.score.common._cuda_graph import CapturedScoringGraph

        graph_module = _KinematicScoringModule(
            self.kin_module,
            self.whole_pose_scoring_module._graph_capturable_module(),
            self._kinematic_scoring.base_dofs,
            self._kinematic_scoring.base_coords,
            self._dof_flat_idx,
            self._atom_idx,
        )
        if mode in ("forward", "both") and "forward" not in self._cuda_graphs:
            self._cuda_graphs["forward"] = _InferenceCUDAGraph(
                graph_module, self.masked_dofs
            )
        if mode in ("forward_backward", "both") and "autograd" not in self._cuda_graphs:
            sample = self.masked_dofs.detach().clone().requires_grad_(True)
            with (
                torch.cuda.device(sample.device),
                torch.enable_grad(),
                warnings.catch_warnings(),
            ):
                # Scoring parameters may retain an AccumulateGrad node on
                # another stream. Suppress only that known warmup warning.
                warnings.filterwarnings(
                    "ignore",
                    message="The AccumulateGrad node's stream does not match",
                )
                self._cuda_graphs["autograd"] = CapturedScoringGraph(
                    graph_module, sample
                )
        return self

    @property
    def full_dofs(self) -> torch.Tensor:
        """All kinematic DOFs at the current movable-DOF values."""
        return self._kinematic_scoring.dofs(self.masked_dofs.detach())

    @property
    def full_coords(self) -> torch.Tensor:
        """Pose-stack coordinates at the current movable-DOF values."""
        with torch.no_grad():
            return self._kinematic_scoring.coords(self.masked_dofs)

    @property
    def flat_coords(self) -> torch.Tensor:
        return self.full_coords.view(-1, 3)

    def _reusable_for(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        fold_forest: FoldForest,
        move_map: MoveMap,
    ) -> bool:
        """Return whether this network can score another pose without rendering."""
        # Check cheap Python identity and shape predicates before any tensor
        # comparison. The normal repeated-topology path shares these tensors,
        # so it avoids both rebuilding ``real_atoms`` and synchronizing CUDA.
        if not (
            score_function is self._score_function
            and self._score_function_versions
            == (
                self._score_function._terms_version,
                self._score_function._options_version,
            )
            and pose_stack.packed_block_types is self.pose_stack.packed_block_types
            and pose_stack.constraint_set is self.pose_stack.constraint_set
            and pose_stack.coords.shape == self.pose_stack.coords.shape
            and pose_stack.inter_residue_connections
            is self.pose_stack.inter_residue_connections
            and pose_stack.inter_block_bondsep is self.pose_stack.inter_block_bondsep
            and self.ff_copy is not None
            and self.mm_copy is not None
            and _same_fold_forest(fold_forest, self.ff_copy)
        ):
            return False

        # The fold forest and move map are mutable, so compare their contents
        # against the copies taken at construction rather than identity.
        if not _same_move_map(move_map, self.mm_copy):
            return False

        return (
            pose_stack.block_type_ind is self.pose_stack.block_type_ind
            or torch.equal(pose_stack.block_type_ind, self.pose_stack.block_type_ind)
        ) and (
            pose_stack.block_coord_offset is self.pose_stack.block_coord_offset
            or torch.equal(
                pose_stack.block_coord_offset, self.pose_stack.block_coord_offset
            )
        )

    def _reset(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        fold_forest: FoldForest,
        move_map: MoveMap,
    ) -> bool:
        """Reset for a compatible pose topology; report whether it was reused."""
        if not self._reusable_for(score_function, pose_stack, fold_forest, move_map):
            return False

        self.pose_stack = pose_stack
        # Copy in place: captured graphs read these buffers by address.
        scoring = self._kinematic_scoring
        with torch.no_grad():
            self.whole_pose_scoring_module.weights.copy_(
                score_function.weights_tensor().unsqueeze(1)
            )
            scoring.base_coords.copy_(pose_stack.coords)
            scoring.base_dofs.copy_(self._dofs_from_coords(pose_stack.coords))
            self.masked_dofs.copy_(scoring.base_dofs.view(-1)[self._dof_flat_idx])
        self.masked_dofs.grad = None
        return True

    def forward(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            graph = self._cuda_graphs.get("autograd")
            if graph is not None:
                # Graph replay reuses its output storage. Optimizers such as
                # LBFGS retain earlier losses, so return owned storage.
                return graph(self.masked_dofs).clone()
        else:
            graph = self._cuda_graphs.get("forward")
            if graph is not None:
                return graph(self.masked_dofs)
        return self._kinematic_scoring(self.masked_dofs)

    def pose_stack_from_dofs(self) -> PoseStack:
        return attrs.evolve(self.pose_stack, coords=self.full_coords)
