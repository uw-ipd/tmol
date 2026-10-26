"""Pluggable scoring and local-minimization backends for docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import attrs
import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import (
    GeneIndex,
    LigandDockingDOFs,
    _block_kinforest_nodes,
    _ligand_kinforest_nodes,
    _sidechain_coordinate_indices,
    build_ligand_docking_dofs,
)
from tmol.kinematics import KinForest
from tmol.kinematics.compiled import (
    forward_only_op,
    forward_with_ht_op,
    inverse_kin,
    kin_deriv_op,
)
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.optimization import LBFGS_Armijo, build_kinforest_network, run_min
from tmol.pose import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.common._cuda_graph import CapturedFunction
from tmol.score.gridscore import GridScorer, GridSet, build_grid
from tmol.score.gridscore.live_hbond import LiveHBondData, build_live_hbond_data
from tmol.score.gridscore.live_ligand_intra import (
    LiveLigandIntraData,
    build_live_ligand_intra_data,
)
from tmol.score.gridscore.live_lk_ball import LiveLKBallData, build_live_lk_ball_data
from tmol.score import ScoreFunction
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_atoms,
    movable_pose_stack,
    resolve_ligand_block,
    sidechain_context_blocks,
    validated_sidechain_blocks,
)


@dataclass(frozen=True)
class MinimizationResult:
    """A minimized pose and its optimizer-authored freed-DOF gene.

    ``genes`` is ``None`` only for backends that do not participate in the GA.
    Continuous minimization always returns the fixed-order jump, ligand-chi,
    and selected-sidechain-chi tensor directly from its final kinematic DOFs.
    """

    pose_stack: PoseStack
    genes: Optional[torch.Tensor]
    masked_dofs: Optional[torch.Tensor] = None


def _genes_from_kinforest_dofs(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    kf_network,
) -> torch.Tensor:
    """Map final optimizer DOFs to six jump DOFs plus logical torsions."""

    kmd = kf_network.kin_module.kmd
    raw_dofs = kf_network.full_dofs.detach().clone()
    raw_dofs.reshape(-1)[kf_network._dof_flat_idx] = kf_network.masked_dofs.detach()
    return _genes_from_raw_kinforest_dofs(
        pose_stack,
        docking_dofs,
        kmd.forest,
        kf_network.dof_mask,
        raw_dofs,
    )


def _genes_from_raw_kinforest_dofs(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    forest: KinForest,
    dof_mask: torch.Tensor,
    raw_dofs: torch.Tensor,
    ligand_nodes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map raw kinematic DOFs directly to the docking gene layout."""

    index = GeneIndex.build(
        pose_stack, docking_dofs, forest, dof_mask, raw_dofs.dtype, ligand_nodes
    )
    return index.genes_from_raw_dofs(raw_dofs).to(dtype=pose_stack.coords.dtype)


class ScoringBackend(ABC):
    """The scoring contract used by the docking loop.

    A future ``GridBackend`` must implement these same two methods. ``score``
    returns one total beta16-compatible energy per pose, with shape
    ``[n_poses]``. ``minimize`` performs one batched local minimization over
    exactly the freedoms described by ``docking_dofs`` and returns a
    :class:`MinimizationResult`. A grid implementation may use a different
    internal objective and gradient representation, but it must preserve these
    input and output semantics so the population loop does not depend on the
    backend.

    A backend is built for one ligand: ``ligand_block`` is its block index in
    the reference complex, and ``sidechain_blocks`` the receptor residues
    (block indices) whose sidechains move with it.
    """

    ligand_block: int
    sidechain_blocks: tuple[int, ...] = ()

    def search_view(
        self, pose_stack: PoseStack
    ) -> tuple[PoseStack, "ScoringBackend", tuple[int, ...]]:
        """The pose, backend and flexible blocks the GA actually searches over."""
        return pose_stack, self, ()

    @abstractmethod
    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return the total energy of each pose as a ``[n_poses]`` tensor."""

    @abstractmethod
    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Minimize and return coordinates plus the final freed-DOF gene."""


class ContinuousBackend(ScoringBackend):
    """Full continuous beta2016 scoring and kinematic minimization.

    ``param_db`` should be the ligand-extended parameter database returned by
    ligand preparation. If it is omitted, tmol's memoized default beta2016
    score function is used. A score function is cached per device, while the
    whole-pose scoring module is rendered for each input PoseStack.

    Args:
        ligand_block: Block index of the ligand in the complex to be docked.
        param_db: Optional ligand-extended scoring parameter database.
        kin_dtype: Internal-coordinate dtype for minimization. By default the
            input coordinate dtype is used.
        value_only_line_search: Evaluate Armijo trials without gradients and
            refresh the gradient at the accepted point (default: False).
    """

    def __init__(
        self,
        ligand_block: int,
        param_db: Optional[ParameterDatabase] = None,
        *,
        kin_dtype: Optional[torch.dtype] = None,
        value_only_line_search: bool = False,
    ) -> None:
        if kin_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("kin_dtype must be torch.float32 or torch.float64")
        self.ligand_block = ligand_block
        self.param_db = param_db
        self.kin_dtype = kin_dtype
        self.value_only_line_search = value_only_line_search
        self._score_functions: dict[torch.device, ScoreFunction] = {}

    def _score_function(self, device: torch.device) -> ScoreFunction:
        device = torch.device(device)
        if device not in self._score_functions:
            self._score_functions[device] = beta2016_score_function(
                device, param_db=self.param_db
            )
        return self._score_functions[device]

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Score a PoseStack with the weighted continuous beta2016 function."""

        sfxn = self._score_function(pose_stack.device)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        with torch.no_grad():
            energies = scorer(pose_stack.coords)
        expected_shape = (pose_stack.n_poses,)
        if energies.shape != expected_shape:
            raise RuntimeError(
                "continuous whole-pose scorer returned shape "
                f"{tuple(energies.shape)}; expected {expected_shape}"
            )
        return energies

    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Run docking-DOF minimization and return optimizer-authored genes."""
        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        kf_network = build_kinforest_network(
            pose_stack,
            self._score_function(pose_stack.device),
            docking_dofs.fold_forest,
            docking_dofs.move_map,
            kin_dtype=kin_dtype,
        )
        if initial_raw_dofs is not None:
            if initial_raw_dofs.shape != kf_network.full_dofs.shape:
                raise ValueError(
                    "initial raw DOF shape does not match the minimizer: "
                    f"{tuple(initial_raw_dofs.shape)} != "
                    f"{tuple(kf_network.full_dofs.shape)}"
                )
            kf_network.full_dofs = initial_raw_dofs.to(dtype=kin_dtype).detach().clone()
            with torch.no_grad():
                kf_network.masked_dofs.copy_(kf_network.full_dofs[kf_network.dof_mask])
        optimizer_kwargs = {"value_only_line_search": self.value_only_line_search}
        if maxiter is not None:
            optimizer_kwargs["max_iter"] = maxiter
        minimized = run_min(
            kf_network,
            optimizer_kwargs=optimizer_kwargs,
        )
        minimized.coords = minimized.coords.detach()
        genes = _genes_from_kinforest_dofs(
            pose_stack, docking_dofs, kf_network
        ).detach()
        return MinimizationResult(
            pose_stack=minimized,
            genes=genes,
            masked_dofs=kf_network.masked_dofs.detach().clone(),
        )


class _GridPoseObjective(torch.nn.Module):
    """Gather one ligand block and score it against a shared receptor grid."""

    def __init__(
        self,
        backend: "GridBackend",
        pose_stack: PoseStack,
        ligand_block: int,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.n_poses = pose_stack.n_poses
        self.pose_indices = torch.arange(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        self.atom_indices = backend._movable_atom_indices(pose_stack, ligand_block)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        ligand_coordinates = coordinates[self.pose_indices[:, None], self.atom_indices]
        return self.backend._score_ligand_coordinates(ligand_coordinates)


class _LightweightGridObjective:
    """Graph-free ligand-grid energy and docking-DOF derivative evaluator."""

    def __init__(
        self,
        backend: "GridBackend",
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        kin_dtype: torch.dtype,
        initial_raw_dofs: Optional[torch.Tensor],
        cuda_graph: bool = False,
    ) -> None:
        self.backend = backend
        self.pose_stack = pose_stack
        self.docking_dofs = docking_dofs
        kin_module = PoseStackKinematicsModule(pose_stack, docking_dofs.fold_forest)
        full_kmd = kin_module.kmd
        self.forest = full_kmd.forest
        self.dof_mask = MinimizerMap(
            pose_stack, full_kmd, docking_dofs.move_map
        ).dof_mask

        if initial_raw_dofs is not None:
            expected_shape = (full_kmd.forest.id.shape[0], 9)
            if initial_raw_dofs.shape != expected_shape:
                raise ValueError(
                    "initial raw DOF shape does not match the minimizer: "
                    f"{tuple(initial_raw_dofs.shape)} != {expected_shape}"
                )
            full_dofs = None
        else:
            kin_coords = torch.zeros(
                (full_kmd.forest.id.shape[0], 3),
                dtype=kin_dtype,
                device=pose_stack.device,
            )
            kin_coords[1:] = pose_stack.coords.reshape(-1, 3)[
                full_kmd.forest.id[1:].to(torch.int64)
            ].to(kin_dtype)
            full_dofs = inverse_kin(
                kin_coords,
                full_kmd.forest.parent,
                full_kmd.forest.frame_x,
                full_kmd.forest.frame_y,
                full_kmd.forest.frame_z,
                full_kmd.forest.doftype,
            )
        ligand_block_type = pose_stack.block_type(0, docking_dofs.ligand_block)
        self.ligand_nodes = _ligand_kinforest_nodes(
            pose_stack,
            docking_dofs.ligand_block,
            ligand_block_type.n_atoms,
            self.forest.id,
        )
        # Reuse the already-validated full-pose ordering. All compiled operations
        # run under no_grad, so retaining fixed receptor nodes costs forward
        # buffers but does not retain an autograd graph across L-BFGS iterations.
        # The full ordering is retained so ligand and selected sidechain
        # coordinate gradients can both be propagated to their freed DOFs.
        self.nodes_f = kin_module.nodes_f
        self.scans_f = kin_module.scans_f
        self.gens_f = kin_module.gens_f
        self.nodes_b = kin_module.nodes_b
        self.scans_b = kin_module.scans_b
        self.gens_b = kin_module.gens_b
        self.kintree = kin_module.kinforest

        self.dof_flat_idx = (
            self.dof_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )
        if initial_raw_dofs is None:
            self.full_dofs = full_dofs.detach().clone()
        else:
            self.full_dofs = initial_raw_dofs.to(dtype=kin_dtype).detach().clone()
        self.initial_masked_dofs = self.full_dofs.reshape(-1)[self.dof_flat_idx].clone()

        self.ligand_atom_indices = backend._ligand_atom_indices(
            pose_stack, docking_dofs.ligand_block
        )
        self.sidechain_nodes, self.sidechain_atom_indices = (
            _sidechain_coordinate_indices(pose_stack, docking_dofs, self.forest)
        )
        if backend.sidechain_blocks:
            movable_node_parts = []
            for block in range(pose_stack.max_n_blocks):
                block_type = pose_stack.block_type(0, block)
                movable_node_parts.append(
                    _block_kinforest_nodes(
                        pose_stack,
                        block,
                        block_type.n_atoms,
                        self.forest.id,
                    )
                )
            self.movable_nodes = torch.cat(movable_node_parts, dim=1)
        else:
            self.movable_nodes = self.ligand_nodes

        pose_for_node = torch.div(
            self.forest.id,
            pose_stack.max_n_pose_atoms,
            rounding_mode="floor",
        )
        pose_for_dof = (
            pose_for_node.unsqueeze(1).expand(self.full_dofs.shape).reshape(-1)
        )
        self.segment_ids = pose_for_dof[self.dof_flat_idx].to(torch.int64)
        if bool(torch.any(self.segment_ids < 0)):
            raise ValueError("root node DOFs cannot be minimized")
        self._captured = (
            CapturedFunction(self._energy_and_gradient, self.initial_masked_dofs)
            if cuda_graph and pose_stack.device.type == "cuda"
            else None
        )

    def _set_masked_dofs(self, masked_dofs: torch.Tensor) -> None:
        if masked_dofs.shape != self.initial_masked_dofs.shape:
            raise ValueError(
                "masked DOFs have shape "
                f"{tuple(masked_dofs.shape)}; expected "
                f"{tuple(self.initial_masked_dofs.shape)}"
            )
        self.full_dofs.reshape(-1).index_copy_(
            0, self.dof_flat_idx, masked_dofs.detach()
        )

    def _forward_only(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        self._set_masked_dofs(masked_dofs)
        return forward_only_op(
            self.full_dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.kintree,
        )

    def energy(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        """Evaluate per-pose energy without constructing an autograd graph."""

        with torch.no_grad():
            kin_coords = self._forward_only(masked_dofs)
            movable_coords = kin_coords[self.movable_nodes]
            return self.backend._score_ligand_coordinates(movable_coords)

    def energy_and_gradient(
        self, masked_dofs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate energy and analytic dE/d(freed raw DOFs), graph-free.

        On CUDA the evaluation is a replay of one captured graph."""

        if self._captured is not None:
            return self._captured(masked_dofs)
        return self._energy_and_gradient(masked_dofs)

    def _energy_and_gradient(
        self, masked_dofs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self._set_masked_dofs(masked_dofs)
            kin_coords, hts = forward_with_ht_op(
                self.full_dofs,
                self.nodes_f,
                self.scans_f,
                self.gens_f,
                self.kintree,
            )
            movable_coords = kin_coords[self.movable_nodes]
            energies, movable_gradients = (
                self.backend._score_ligand_coordinates_with_gradients(movable_coords)
            )
            coordinate_gradients = torch.zeros_like(kin_coords)
            coordinate_gradients[self.movable_nodes.reshape(-1)] = (
                movable_gradients.reshape(-1, 3)
            )
            dof_gradients = kin_deriv_op(
                coordinate_gradients,
                hts,
                self.full_dofs,
                self.nodes_b,
                self.scans_b,
                self.gens_b,
                self.kintree,
            )
            masked_gradients = dof_gradients.reshape(-1)[self.dof_flat_idx]
        return energies, masked_gradients

    def final_result(self, masked_dofs: torch.Tensor) -> MinimizationResult:
        """Build output coordinates and gene directly from final raw DOFs."""

        with torch.no_grad():
            kin_coords = self._forward_only(masked_dofs)
            full_coords = self.pose_stack.coords.detach().clone()
            pose_indices = torch.arange(
                self.pose_stack.n_poses,
                dtype=torch.int64,
                device=self.pose_stack.device,
            )
            full_coords[pose_indices[:, None], self.ligand_atom_indices] = kin_coords[
                self.ligand_nodes
            ].to(full_coords.dtype)
            if self.sidechain_nodes.shape[1]:
                full_coords[pose_indices[:, None], self.sidechain_atom_indices] = (
                    kin_coords[self.sidechain_nodes].to(full_coords.dtype)
                )
            minimized_pose = attrs.evolve(self.pose_stack, coords=full_coords)
            genes = _genes_from_raw_kinforest_dofs(
                self.pose_stack,
                self.docking_dofs,
                self.forest,
                self.dof_mask,
                self.full_dofs,
                self.ligand_nodes,
            ).detach()
        return MinimizationResult(
            pose_stack=minimized_pose,
            genes=genes,
            masked_dofs=masked_dofs.detach().clone(),
        )


class GridBackend(ScoringBackend):
    """Rigid-receptor grid plus live hbond, intra, and LK-ball scoring.

    One :class:`GridSet` is retained by the backend and shared by every pose.
    Device/dtype copies are cached once and do not scale with population size.
    The grid contains fa_atr, fa_rep, fa_sol, and fa_elec, plus three weighted
    LK-ball grid families when ``lk_ball_grid`` is true. One receptor hbond
    structure, ligand-intra topology, and ligand-water topology are shared by
    every ligand pose; ``lk_ball_grid=False`` retains the exact live LK-ball
    implementation. Ligand-intra search weights match Rosetta GridScorer's
    softened ``oneb_wtd_`` objective.

    ``value_only_line_search=True`` evaluates Armijo trials with energy only
    and refreshes analytic gradients at accepted points (default: False).
    """

    def __init__(
        self,
        grids: GridSet,
        ligand_block: int,
        ligand_atom_types: tuple[str, ...],
        ligand_charges: torch.Tensor,
        param_db: Optional[ParameterDatabase] = None,
        *,
        kin_dtype: Optional[torch.dtype] = None,
        live_hbond: Optional[LiveHBondData] = None,
        live_ligand_intra: Optional[LiveLigandIntraData] = None,
        live_lk_ball: Optional[LiveLKBallData] = None,
        lk_ball_grid: bool = False,
        sidechain_blocks: tuple[int, ...] = (),
        context_blocks: tuple[int, ...] = (),
        grid_atom_indices: Optional[torch.Tensor] = None,
        movable_n_atoms: Optional[int] = None,
        value_only_line_search: bool = False,
        cuda_graph: bool = True,
    ) -> None:
        self.grids = grids
        self.ligand_block = ligand_block
        self.ligand_atom_types = tuple(ligand_atom_types)
        self.ligand_charges = ligand_charges.detach().to("cpu", torch.float64)
        self.param_db = param_db
        self.kin_dtype = kin_dtype
        self.live_hbond = live_hbond
        self.live_ligand_intra = live_ligand_intra
        self.live_lk_ball = live_lk_ball
        self.lk_ball_grid = lk_ball_grid
        self.sidechain_blocks = tuple(sidechain_blocks)
        self.context_blocks = tuple(context_blocks)
        self.grid_atom_indices = grid_atom_indices
        self.movable_n_atoms = movable_n_atoms or len(ligand_atom_types)
        self.value_only_line_search = value_only_line_search
        self.cuda_graph = cuda_graph
        self._scorers: dict[tuple[torch.device, torch.dtype], GridScorer] = {}
        self._atom_index_cache: dict[tuple, torch.Tensor] = {}

    def search_view(
        self, pose_stack: PoseStack
    ) -> tuple[PoseStack, "GridBackend", tuple[int, ...]]:
        """Compact ligand(+flexible sidechain) pose the GA searches over."""
        compact_sidechains: tuple[int, ...] = ()
        if self.sidechain_blocks:
            _, compact_sidechains, _ = movable_block_mapping(
                pose_stack, self.ligand_block, self.sidechain_blocks
            )
        compact = movable_pose_stack(
            pose_stack, self.ligand_block, self.sidechain_blocks
        )
        return compact, self.for_ligand_block(0), compact_sidechains

    def for_ligand_block(self, ligand_block: int) -> "GridBackend":
        """Return a lightweight view targeting another ligand block index.

        Grid construction fixes the ligand chemistry, not its block position in
        a scoring PoseStack. The ligand-only docking representation places the
        same block type at index zero. A shallow copy preserves the single grid
        and its device/dtype caches while allowing that representation to use
        the ordinary backend contract, whose ``score`` method has no block
        argument.
        """
        result = copy.copy(self)
        result.ligand_block = ligand_block
        return result

    @classmethod
    def from_pose_stack(
        cls,
        reference_pose_stack: PoseStack,
        ligand,
        param_db: ParameterDatabase,
        *,
        flexible_residues: Sequence[int] | None = None,
        voxel_spacing: float = 0.25,
        bbox_padding: float = 4.0,
        kin_dtype: Optional[torch.dtype] = None,
        lk_ball_grid: bool = True,
        include_ligand_intra: bool = True,
        value_only_line_search: bool = False,
        cuda_graph: bool = True,
        ligand_intra_weight_profile: str = "search",
    ) -> "GridBackend":
        """Build the receptor grid and live terms for one ligand of a complex.

        Args:
            reference_pose_stack: One CPU pose holding the receptor and the
                ligand (in any placement; the search starts from random poses).
            ligand: The ligand: residue name (``"ATP"``), block index, or a
                boolean mask / atom indices selecting its block (see
                :func:`resolve_ligand_block`).
            param_db: Scoring database including the ligand's parameters.
            flexible_residues: Receptor residues (block indices, e.g. from
                :func:`select_pocket_residues`) whose sidechains move.

        ``lk_ball_grid=True`` selects Rosetta's weighted gridded approximation;
        ``False`` retains the exact live LK-ball path. Ligand-intra scoring is
        enabled by default with Rosetta's softened search profile.
        ``value_only_line_search`` opts into energy-only Armijo trials.
        ``cuda_graph`` replays each minimization's energy/gradient evaluation
        from a captured CUDA graph (ignored on CPU).
        """

        if reference_pose_stack.n_poses != 1:
            raise ValueError("grid construction requires one reference pose")
        if reference_pose_stack.device.type != "cpu":
            raise ValueError("grid construction requires a CPU reference pose")
        ligand_block = resolve_ligand_block(reference_pose_stack, ligand)
        pbt = reference_pose_stack.packed_block_types
        block_type_index = int(reference_pose_stack.block_type_ind64[0, ligand_block])
        block_type = pbt.active_block_types[block_type_index]
        sidechain_blocks = validated_sidechain_blocks(
            reference_pose_stack, ligand_block, flexible_residues or ()
        )
        context_blocks = sidechain_context_blocks(
            reference_pose_stack, ligand_block, sidechain_blocks
        )
        moving = movable_atoms(reference_pose_stack, ligand_block, sidechain_blocks)

        # Compact-pose atoms the GA moves: the ligand, plus flexible chi atoms.
        grid_atom_indices = None
        if sidechain_blocks:
            movable_pose = movable_pose_stack(
                reference_pose_stack, ligand_block, sidechain_blocks
            )
            _, compact_sidechains, _ = movable_block_mapping(
                reference_pose_stack, ligand_block, sidechain_blocks
            )
            movable_dofs = build_ligand_docking_dofs(
                movable_pose, 0, sidechain_blocks=compact_sidechains
            )
            _, sidechain_atom_indices = _sidechain_coordinate_indices(
                movable_pose,
                movable_dofs,
                PoseStackKinematicsModule(
                    movable_pose, movable_dofs.fold_forest
                ).kmd.forest,
            )
            grid_atom_indices = torch.cat(
                (
                    torch.arange(block_type.n_atoms, dtype=torch.int64),
                    sidechain_atom_indices[0].to(device="cpu"),
                )
            ).contiguous()
        query = (
            torch.arange(block_type.n_atoms)
            if grid_atom_indices is None
            else grid_atom_indices
        )
        source_types = [
            int(reference_pose_stack.block_type_ind64[0, block])
            for block in moving.source_blocks
        ]
        atom_types = [
            atom.atom_type
            for index in source_types
            for atom in pbt.active_block_types[index].atoms[
                : pbt.active_block_types[index].n_atoms
            ]
        ]
        ligand_atom_types = tuple(atom_types[i] for i in query.tolist())

        grids = build_grid(
            reference_pose_stack,
            tuple(sorted(set(ligand_atom_types))),
            param_db,
            ligand_name3=block_type.name3,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
            build_lk_ball=lk_ball_grid,
            excluded_receptor_blocks=sidechain_blocks + context_blocks,
        )
        # build_grid initializes the packed ljlk/elec parameters, incl. charges.
        ligand_charges = torch.cat(
            [
                pbt.elec_partial_charge[index, : pbt.active_block_types[index].n_atoms]
                for index in source_types
            ]
        )[query]
        live_hbond = build_live_hbond_data(
            reference_pose_stack,
            ligand_block,
            param_db,
            sidechain_blocks=sidechain_blocks,
            movable_atoms=grid_atom_indices,
        )
        live_ligand_intra = (
            build_live_ligand_intra_data(
                reference_pose_stack,
                ligand_block,
                param_db,
                weight_profile=ligand_intra_weight_profile,
                sidechain_blocks=sidechain_blocks,
            )
            if include_ligand_intra
            else None
        )
        live_lk_ball = build_live_lk_ball_data(
            reference_pose_stack,
            ligand_block,
            param_db,
            sidechain_blocks=sidechain_blocks,
            movable_atoms=grid_atom_indices,
        )
        return cls(
            grids,
            ligand_block,
            ligand_atom_types,
            ligand_charges,
            param_db,
            kin_dtype=kin_dtype,
            live_hbond=live_hbond,
            live_ligand_intra=live_ligand_intra,
            live_lk_ball=live_lk_ball,
            lk_ball_grid=lk_ball_grid,
            sidechain_blocks=sidechain_blocks,
            context_blocks=context_blocks,
            grid_atom_indices=grid_atom_indices,
            movable_n_atoms=moving.n_atoms,
            value_only_line_search=value_only_line_search,
            cuda_graph=cuda_graph,
        )

    def _scorer(self, device: torch.device, dtype: torch.dtype) -> GridScorer:
        """The grid scorer for one device and precision, built on first use."""
        key = (torch.device(device), dtype)
        if key not in self._scorers:
            self._scorers[key] = GridScorer(
                self.grids,
                self.ligand_atom_types,
                self.ligand_charges,
                device=device,
                dtype=dtype,
                score_function=beta2016_score_function(
                    torch.device(device), param_db=self.param_db
                ),
                live_hbond=self.live_hbond,
                live_lk_ball=self.live_lk_ball,
                live_ligand_intra=self.live_ligand_intra,
                lk_ball_grid=self.lk_ball_grid,
                grid_atoms=self.grid_atom_indices,
            )
        return self._scorers[key]

    def _ligand_atom_indices(
        self, pose_stack: PoseStack, ligand_block: int
    ) -> torch.Tensor:
        if ligand_block != self.ligand_block:
            raise ValueError(
                f"backend was built for ligand block {self.ligand_block}, "
                f"not {ligand_block}"
            )
        cache_key = (
            torch.device(pose_stack.device),
            pose_stack.n_poses,
            pose_stack.max_n_pose_atoms,
            pose_stack.max_n_blocks,
            ligand_block,
            id(pose_stack.packed_block_types),
        )
        cached = self._atom_index_cache.get(cache_key)
        if cached is not None:
            return cached

        block_types = pose_stack.block_type_ind64[:, ligand_block]
        if bool(torch.any(block_types < 0)):
            raise ValueError("ligand block is absent from at least one pose")
        first_type = block_types[0]
        if not bool(torch.all(block_types == first_type)):
            raise ValueError("GridBackend requires one homogeneous ligand type")
        block_type = pose_stack.packed_block_types.active_block_types[int(first_type)]
        if not self.sidechain_blocks and block_type.n_atoms != len(
            self.ligand_atom_types
        ):
            raise ValueError("pose ligand atom count does not match the grid backend")
        offsets = pose_stack.block_coord_offset64[:, ligand_block, None]
        local_atoms = torch.arange(
            block_type.n_atoms, dtype=torch.int64, device=pose_stack.device
        )
        atom_indices = offsets + local_atoms[None, :]
        self._atom_index_cache[cache_key] = atom_indices
        return atom_indices

    def _movable_atom_indices(
        self, pose_stack: PoseStack, ligand_block: int
    ) -> torch.Tensor:
        """Return compact whole-movable coordinate indices for every pose."""

        if not self.sidechain_blocks:
            return self._ligand_atom_indices(pose_stack, ligand_block)
        expected_blocks = 1 + len(self.sidechain_blocks) + len(self.context_blocks)
        if ligand_block != 0 or pose_stack.max_n_blocks != expected_blocks:
            raise ValueError(
                "flexible GridBackend requires a compact ligand-plus-sidechains pose"
            )
        if pose_stack.max_n_pose_atoms != self.movable_n_atoms:
            raise ValueError(
                "movable pose atom count does not match the grid backend topology"
            )
        return torch.arange(
            self.movable_n_atoms,
            dtype=torch.int64,
            device=pose_stack.device,
        )[None, :].expand(pose_stack.n_poses, -1)

    def _score_ligand_coordinates(
        self, ligand_coordinates: torch.Tensor
    ) -> torch.Tensor:
        scorer = self._scorer(ligand_coordinates.device, ligand_coordinates.dtype)
        return scorer(ligand_coordinates)

    def _score_ligand_coordinates_with_gradients(
        self, ligand_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scorer = self._scorer(ligand_coordinates.device, ligand_coordinates.dtype)
        return scorer.value_and_grad(ligand_coordinates)

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return one grid GA objective energy per pose.

        This is the grid ligand-receptor terms (grid LJ/LK/elec + live hbond +
        live LK-ball) plus, by default, the live ligand-intramolecular energy --
        NOT a pure interface energy. Disable the intra contribution by building
        the backend with ``include_ligand_intra=False``.
        """

        atom_indices = self._movable_atom_indices(pose_stack, self.ligand_block)
        pose_indices = torch.arange(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        ligand_coordinates = pose_stack.coords[pose_indices[:, None], atom_indices]
        with torch.no_grad():
            energies = self._score_ligand_coordinates(ligand_coordinates)
        expected_shape = (pose_stack.n_poses,)
        if energies.shape != expected_shape:
            raise RuntimeError(
                f"grid scorer returned {tuple(energies.shape)}; "
                f"expected {expected_shape}"
            )
        return energies

    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Minimize with direct analytic grid and kinematic derivatives.

        The optimizer holds only freed docking DOFs and a bounded L-BFGS
        history. Each closure uses the valid full-pose KinForest ordering,
        short-lived compiled kinematic/grid-gradient buffers, and explicitly
        assigned gradients. Ligand-intra kernels use a local autograd graph
        internally and return detached values; no graph reaches the optimizer.
        Selected receptor-sidechain DOFs receive grid, live-interface, and
        whole-movable gradients; all other receptor DOFs remain fixed.
        """

        if docking_dofs.ligand_block != self.ligand_block:
            raise ValueError("docking DOFs target a different ligand block")
        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        objective = _LightweightGridObjective(
            self,
            pose_stack,
            docking_dofs,
            kin_dtype,
            initial_raw_dofs,
            cuda_graph=self.cuda_graph,
        )
        masked_dofs = torch.nn.Parameter(
            objective.initial_masked_dofs, requires_grad=True
        )
        max_iterations = 200 if maxiter is None else maxiter
        optimizer = LBFGS_Armijo(
            [masked_dofs],
            max_iter=max_iterations,
            # Ten pairs are ample for the 6+chi ligand system and bound the
            # per-pose optimizer state independently of the iteration budget.
            history_size=min(10, max_iterations),
            segment_ids=objective.segment_ids,
            value_only_line_search=self.value_only_line_search,
        )

        def closure():
            energies, gradients = objective.energy_and_gradient(masked_dofs)
            if masked_dofs.grad is None:
                masked_dofs.grad = torch.empty_like(masked_dofs)
            masked_dofs.grad.copy_(gradients)
            return energies

        def value_only_closure():
            return objective.energy(masked_dofs)

        if self.value_only_line_search:
            optimizer.step(closure, value_only_closure)
        else:
            optimizer.step(closure)
        return objective.final_result(masked_dofs.detach())
