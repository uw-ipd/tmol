"""Pluggable scoring and local-minimization backends for docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional

import attrs
import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    _ligand_kinforest_nodes,
    _wrap_angle,
)
from tmol.kinematics.datatypes import KinForest, JumpDOFTypes, NodeType
from tmol.kinematics.compiled import (
    forward_only_op,
    forward_with_ht_op,
    inverse_kin,
    kin_deriv_op,
)
from tmol.kinematics.move_map import MinimizerMap
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.minimizers import build_kinforest_network, run_min
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.gridscore import (
    GridSet,
    LiveHBondData,
    LiveLKBallData,
    build_grid,
    build_live_hbond_data,
    build_live_lk_ball_data,
    probe_atom_types_from_name3,
    score_ligand_grid_batched,
    score_ligand_grid_batched_with_gradients,
)
from tmol.score.score_function import ScoreFunction


@dataclass(frozen=True)
class MinimizationResult:
    """A minimized pose and its optimizer-authored freed-DOF gene.

    ``genes`` is ``None`` only for backends that do not participate in the GA.
    Continuous minimization always returns the fixed-order six-jump-plus-chi
    tensor directly from its final kinematic DOFs.
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
    raw_dofs.reshape(-1)[kf_network._dof_flat_idx] = (
        kf_network.masked_dofs.detach()
    )
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

    jump_mask = (
        (forest.doftype == int(NodeType.jump))
        & torch.all(
            dof_mask[:, : int(JumpDOFTypes.RBdel_gamma) + 1],
            dim=1,
        )
    )
    jump_node_list = torch.nonzero(jump_mask, as_tuple=False).flatten()
    if jump_node_list.numel() != pose_stack.n_poses:
        raise ValueError("expected exactly one movable ligand jump per pose")
    jump_pose = torch.div(
        forest.id[jump_node_list],
        pose_stack.max_n_pose_atoms,
        rounding_mode="floor",
    ).to(torch.int64)
    jump_nodes = torch.empty(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    jump_nodes.scatter_(0, jump_pose, jump_node_list)
    jump_dofs = raw_dofs[
        jump_nodes[:, None],
        torch.arange(6, dtype=torch.int64, device=pose_stack.device)[None, :],
    ]

    block_type_index = int(
        pose_stack.block_type_ind64[0, docking_dofs.ligand_block]
    )
    block_type = pose_stack.packed_block_types.active_block_types[
        block_type_index
    ]
    if ligand_nodes is None:
        ligand_nodes = _ligand_kinforest_nodes(
            pose_stack,
            docking_dofs.ligand_block,
            block_type.n_atoms,
            forest.id,
        )
    raw_atoms = torch.tensor(
        [mapping.raw_atom for mapping in docking_dofs.torsions],
        dtype=torch.int64,
        device=pose_stack.device,
    )
    dof_types = torch.tensor(
        [int(mapping.dof_type) for mapping in docking_dofs.torsions],
        dtype=torch.int64,
        device=pose_stack.device,
    )
    signs = torch.tensor(
        [mapping.sign for mapping in docking_dofs.torsions],
        dtype=raw_dofs.dtype,
        device=pose_stack.device,
    )
    if docking_dofs.torsions:
        offsets = torch.stack(
            [mapping.offset for mapping in docking_dofs.torsions], dim=1
        ).to(dtype=raw_dofs.dtype)
    else:
        offsets = torch.empty(
            (pose_stack.n_poses, 0),
            dtype=raw_dofs.dtype,
            device=pose_stack.device,
        )
    raw_torsions = raw_dofs[
        ligand_nodes[:, raw_atoms], dof_types[None, :]
    ]
    logical_torsions = _wrap_angle(
        signs[None, :] * raw_torsions + offsets
    )
    return torch.cat((jump_dofs, logical_torsions), dim=1).to(
        dtype=pose_stack.coords.dtype
    )


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
    """

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
        param_db: Optional ligand-extended scoring parameter database.
        kin_dtype: Internal-coordinate dtype for minimization. By default the
            input coordinate dtype is used.
    """

    def __init__(
        self,
        param_db: Optional[ParameterDatabase] = None,
        *,
        kin_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if kin_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("kin_dtype must be torch.float32 or torch.float64")
        self.param_db = param_db
        self.kin_dtype = kin_dtype
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
        """Run ligand-only minimization and return optimizer-authored genes."""

        if maxiter is not None and (
            not isinstance(maxiter, int)
            or isinstance(maxiter, bool)
            or maxiter < 1
        ):
            raise ValueError("maxiter must be a positive integer")
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
            if initial_raw_dofs.device != pose_stack.device:
                raise ValueError(
                    "initial raw DOFs and pose stack must be on the same device"
                )
            kf_network.full_dofs = initial_raw_dofs.to(
                dtype=kin_dtype
            ).detach().clone()
            with torch.no_grad():
                kf_network.masked_dofs.copy_(
                    kf_network.full_dofs[kf_network.dof_mask]
                )
        minimized = run_min(
            kf_network,
            optimizer_kwargs=(
                None if maxiter is None else {"max_iter": maxiter}
            ),
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
        self.atom_indices = backend._ligand_atom_indices(
            pose_stack, ligand_block
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        ligand_coordinates = coordinates[
            self.pose_indices[:, None], self.atom_indices
        ]
        return self.backend._score_ligand_coordinates(ligand_coordinates)


class _LightweightGridObjective:
    """Graph-free grid energy and ligand-DOF derivative evaluator."""

    def __init__(
        self,
        backend: "GridBackend",
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        kin_dtype: torch.dtype,
        initial_raw_dofs: Optional[torch.Tensor],
    ) -> None:
        self.backend = backend
        self.pose_stack = pose_stack
        self.docking_dofs = docking_dofs
        kin_module = PoseStackKinematicsModule(
            pose_stack, docking_dofs.fold_forest
        )
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
            if initial_raw_dofs.device != pose_stack.device:
                raise ValueError(
                    "initial raw DOFs and pose stack must be on the same device"
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
        self.ligand_nodes = _ligand_kinforest_nodes(
            pose_stack,
            docking_dofs.ligand_block,
            len(backend.ligand_atom_types),
            self.forest.id,
        )
        # Reuse the already-validated full-pose ordering. Only ligand DOFs are
        # movable, and all compiled operations run under no_grad, so retaining
        # receptor nodes costs forward buffers but does not retain an autograd
        # graph across L-BFGS iterations.
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
            self.full_dofs = (
                initial_raw_dofs.to(dtype=kin_dtype).detach().clone()
            )
        self.initial_masked_dofs = self.full_dofs.reshape(-1)[
            self.dof_flat_idx
        ].clone()

        self.ligand_atom_indices = backend._ligand_atom_indices(
            pose_stack, docking_dofs.ligand_block
        )

        pose_for_node = torch.div(
            self.forest.id,
            pose_stack.max_n_pose_atoms,
            rounding_mode="floor",
        )
        pose_for_dof = pose_for_node.unsqueeze(1).expand(
            self.full_dofs.shape
        ).reshape(-1)
        self.segment_ids = pose_for_dof[self.dof_flat_idx].to(torch.int64)
        if bool(torch.any(self.segment_ids < 0)):
            raise ValueError("root node DOFs cannot be minimized")

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
            ligand_coords = kin_coords[self.ligand_nodes]
            return self.backend._score_ligand_coordinates(ligand_coords)

    def energy_and_gradient(
        self, masked_dofs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate energy and analytic dE/d(freed raw DOFs), graph-free."""

        with torch.no_grad():
            self._set_masked_dofs(masked_dofs)
            kin_coords, hts = forward_with_ht_op(
                self.full_dofs,
                self.nodes_f,
                self.scans_f,
                self.gens_f,
                self.kintree,
            )
            ligand_coords = kin_coords[self.ligand_nodes]
            energies, ligand_gradients = (
                self.backend._score_ligand_coordinates_with_gradients(
                    ligand_coords
                )
            )
            coordinate_gradients = torch.zeros_like(kin_coords)
            coordinate_gradients[self.ligand_nodes.reshape(-1)] = (
                ligand_gradients.reshape(-1, 3)
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

    def final_result(
        self, masked_dofs: torch.Tensor
    ) -> MinimizationResult:
        """Build output coordinates and gene directly from final raw DOFs."""

        with torch.no_grad():
            kin_coords = self._forward_only(masked_dofs)
            full_coords = self.pose_stack.coords.detach().clone()
            pose_indices = torch.arange(
                self.pose_stack.n_poses,
                dtype=torch.int64,
                device=self.pose_stack.device,
            )
            full_coords[
                pose_indices[:, None], self.ligand_atom_indices
            ] = kin_coords[self.ligand_nodes].to(full_coords.dtype)
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
    """Rigid-receptor grid plus live hbond and selectable LK-ball scoring.

    One :class:`GridSet` is retained by the backend and shared by every pose.
    Device/dtype copies are cached once and do not scale with population size.
    The grid contains fa_atr, fa_rep, fa_sol, and fa_elec, plus three weighted
    LK-ball grid families when ``lk_ball_grid`` is true. One receptor hbond
    structure and ligand-water topology are shared by every ligand pose;
    ``lk_ball_grid=False`` retains the exact live LK-ball implementation.
    Ligand-internal terms remain absent.
    """

    minimizer_kind = "analytic-grid-live-hbond-lkball-f1f2-no-autograd"

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
        live_lk_ball: Optional[LiveLKBallData] = None,
        lk_ball_grid: bool = False,
    ) -> None:
        if not isinstance(grids, GridSet):
            raise TypeError("grids must be a GridSet")
        if not isinstance(ligand_block, int) or isinstance(ligand_block, bool):
            raise TypeError("ligand_block must be an integer")
        if not ligand_atom_types:
            raise ValueError("at least one ligand atom type is required")
        if ligand_charges.ndim != 1 or ligand_charges.shape[0] != len(
            ligand_atom_types
        ):
            raise ValueError(
                "ligand_charges must have one entry per ligand atom type entry"
            )
        if kin_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("kin_dtype must be torch.float32 or torch.float64")
        if not isinstance(lk_ball_grid, bool):
            raise TypeError("lk_ball_grid must be bool")
        unknown_types = sorted(
            set(ligand_atom_types) - set(grids.atom_type_names)
        )
        if unknown_types:
            raise ValueError(
                f"ligand atom types are absent from the grid: {unknown_types}"
            )
        self.grids = grids
        self.ligand_block = ligand_block
        self.ligand_atom_types = tuple(ligand_atom_types)
        self.ligand_charges = ligand_charges.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        self.param_db = param_db
        self.kin_dtype = kin_dtype
        if live_hbond is not None and not isinstance(live_hbond, LiveHBondData):
            raise TypeError("live_hbond must be LiveHBondData or None")
        if live_lk_ball is not None and not isinstance(live_lk_ball, LiveLKBallData):
            raise TypeError("live_lk_ball must be LiveLKBallData or None")
        self.live_hbond = live_hbond
        self.live_lk_ball = live_lk_ball
        self.lk_ball_grid = lk_ball_grid
        if lk_ball_grid and any(
            value is None
            for value in (grids.lk_ball, grids.lk_ball_water, grids.lk_bridge)
        ):
            raise ValueError("lk_ball_grid=True requires all LK-ball grid families")
        if lk_ball_grid and live_lk_ball is None:
            raise ValueError("lk_ball_grid=True requires ligand water topology")
        self._grid_cache: dict[tuple[torch.device, torch.dtype], GridSet] = {}
        self._hbond_cache: dict[tuple[torch.device, torch.dtype], LiveHBondData] = {}
        self._lk_ball_cache: dict[tuple[torch.device, torch.dtype], LiveLKBallData] = {}
        self._charge_cache: dict[
            tuple[torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._score_functions: dict[torch.device, ScoreFunction] = {}
        self._atom_index_cache: dict[
            tuple[torch.device, int, int, int, int, int], torch.Tensor
        ] = {}

    def for_ligand_block(self, ligand_block: int) -> "GridBackend":
        """Return a lightweight view targeting another ligand block index.

        Grid construction fixes the ligand chemistry, not its block position in
        a scoring PoseStack. The ligand-only docking representation places the
        same block type at index zero. A shallow copy preserves the single grid
        and its device/dtype caches while allowing that representation to use
        the ordinary backend contract, whose ``score`` method has no block
        argument.
        """

        if not isinstance(ligand_block, int) or isinstance(ligand_block, bool):
            raise TypeError("ligand_block must be an integer")
        if ligand_block < 0:
            raise ValueError("ligand_block must be nonnegative")
        result = copy.copy(self)
        result.ligand_block = ligand_block
        return result

    @classmethod
    def from_pose_stack(
        cls,
        reference_pose_stack: PoseStack,
        ligand_block: int,
        param_db: ParameterDatabase,
        *,
        ligand_name3: Optional[str] = None,
        voxel_spacing: float = 0.25,
        bbox_padding: float = 4.0,
        kin_dtype: Optional[torch.dtype] = None,
        lk_ball_grid: bool = True,
    ) -> "GridBackend":
        """Build one CPU receptor grid/live-term structure and reusable backend.

        ``lk_ball_grid=True`` selects Rosetta's weighted gridded approximation;
        ``False`` retains the exact live LK-ball path.
        """

        if reference_pose_stack.n_poses != 1:
            raise ValueError("grid construction requires one reference pose")
        if reference_pose_stack.device.type != "cpu":
            raise ValueError("grid construction requires a CPU reference pose")
        if ligand_block < 0 or ligand_block >= reference_pose_stack.max_n_blocks:
            raise ValueError("ligand_block is out of range")
        block_type_index = int(
            reference_pose_stack.block_type_ind64[0, ligand_block]
        )
        if block_type_index < 0:
            raise ValueError("ligand_block is absent from the reference pose")
        pbt = reference_pose_stack.packed_block_types
        block_type = pbt.active_block_types[block_type_index]
        ligand_name3 = ligand_name3 or block_type.name3
        if block_type.name3 != ligand_name3:
            raise ValueError("ligand_name3 does not match ligand_block")

        probe_names = probe_atom_types_from_name3(
            reference_pose_stack, ligand_name3
        )
        grids = build_grid(
            reference_pose_stack,
            probe_names,
            param_db,
            ligand_name3=ligand_name3,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
            build_lk_ball=lk_ball_grid,
        )
        live_hbond = build_live_hbond_data(reference_pose_stack, ligand_block, param_db)
        live_lk_ball = build_live_lk_ball_data(
            reference_pose_stack, ligand_block, param_db
        )
        # build_grid initializes packed ljlk/elec parameters, including charges.
        n_atoms = block_type.n_atoms
        ligand_atom_types = tuple(
            atom.atom_type for atom in block_type.atoms[:n_atoms]
        )
        ligand_charges = pbt.elec_partial_charge[
            block_type_index, :n_atoms
        ]
        return cls(
            grids,
            ligand_block,
            ligand_atom_types,
            ligand_charges,
            param_db,
            kin_dtype=kin_dtype,
            live_hbond=live_hbond,
            live_lk_ball=live_lk_ball,
            lk_ball_grid=lk_ball_grid,
        )

    def _score_function(self, device: torch.device) -> ScoreFunction:
        device = torch.device(device)
        if device not in self._score_functions:
            self._score_functions[device] = beta2016_score_function(
                device, param_db=self.param_db
            )
        return self._score_functions[device]

    def _grid_for(self, device: torch.device, dtype: torch.dtype) -> GridSet:
        key = (torch.device(device), dtype)
        if key not in self._grid_cache:
            self._grid_cache[key] = self.grids.to(device, dtype=dtype)
        return self._grid_cache[key]

    def _charges_for(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (torch.device(device), dtype)
        if key not in self._charge_cache:
            self._charge_cache[key] = self.ligand_charges.to(
                device=device, dtype=dtype
            )
        return self._charge_cache[key]

    def _hbond_for(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[LiveHBondData]:
        if self.live_hbond is None:
            return None
        key = (torch.device(device), dtype)
        if key not in self._hbond_cache:
            self._hbond_cache[key] = self.live_hbond.to(device, dtype)
        return self._hbond_cache[key]

    def _lk_ball_for(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[LiveLKBallData]:
        if self.live_lk_ball is None:
            return None
        key = (torch.device(device), dtype)
        if key not in self._lk_ball_cache:
            self._lk_ball_cache[key] = self.live_lk_ball.to(device, dtype)
        return self._lk_ball_cache[key]

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
        block_type = pose_stack.packed_block_types.active_block_types[
            int(first_type)
        ]
        if block_type.n_atoms != len(self.ligand_atom_types):
            raise ValueError("pose ligand atom count does not match the grid backend")
        offsets = pose_stack.block_coord_offset64[:, ligand_block, None]
        local_atoms = torch.arange(
            block_type.n_atoms, dtype=torch.int64, device=pose_stack.device
        )
        atom_indices = offsets + local_atoms[None, :]
        self._atom_index_cache[cache_key] = atom_indices
        return atom_indices

    def _score_ligand_coordinates(
        self, ligand_coordinates: torch.Tensor
    ) -> torch.Tensor:
        return score_ligand_grid_batched(
            self._grid_for(ligand_coordinates.device, ligand_coordinates.dtype),
            ligand_coordinates,
            self.ligand_atom_types,
            self._charges_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            score_function=self._score_function(ligand_coordinates.device),
            live_hbond=self._hbond_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            live_lk_ball=self._lk_ball_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            lk_ball_grid=self.lk_ball_grid,
        )

    def _score_ligand_coordinates_with_gradients(
        self, ligand_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return score_ligand_grid_batched_with_gradients(
            self._grid_for(ligand_coordinates.device, ligand_coordinates.dtype),
            ligand_coordinates,
            self.ligand_atom_types,
            self._charges_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            score_function=self._score_function(ligand_coordinates.device),
            live_hbond=self._hbond_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            live_lk_ball=self._lk_ball_for(
                ligand_coordinates.device, ligand_coordinates.dtype
            ),
            lk_ball_grid=self.lk_ball_grid,
        )

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return one grid-plus-live-term interface energy per pose."""

        atom_indices = self._ligand_atom_indices(
            pose_stack, self.ligand_block
        )
        pose_indices = torch.arange(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        ligand_coordinates = pose_stack.coords[
            pose_indices[:, None], atom_indices
        ]
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

        Unlike continuous minimization, this path never invokes autograd. The
        optimizer holds only freed ligand DOFs and a bounded L-BFGS history;
        each closure uses the valid full-pose KinForest ordering, short-lived
        compiled kinematic and grid-gradient buffers, and explicitly assigned
        gradients. Receptor DOFs remain fixed.
        """

        if docking_dofs.ligand_block != self.ligand_block:
            raise ValueError("docking DOFs target a different ligand block")
        if maxiter is not None and (
            not isinstance(maxiter, int)
            or isinstance(maxiter, bool)
            or maxiter < 1
        ):
            raise ValueError("maxiter must be a positive integer")
        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        objective = _LightweightGridObjective(
            self,
            pose_stack,
            docking_dofs,
            kin_dtype,
            initial_raw_dofs,
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
        )

        def closure():
            energies, gradients = objective.energy_and_gradient(masked_dofs)
            if masked_dofs.grad is None:
                masked_dofs.grad = torch.empty_like(masked_dofs)
            masked_dofs.grad.copy_(gradients)
            return energies

        optimizer.step(closure)
        return objective.final_result(masked_dofs.detach())
