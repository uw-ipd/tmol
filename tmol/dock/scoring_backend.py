"""Pluggable scoring and local-minimization backends for docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    _ligand_kinforest_nodes,
    _wrap_angle,
)
from tmol.kinematics.datatypes import JumpDOFTypes, NodeType
from tmol.optimization.minimizers import build_kinforest_network, run_min
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
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
    jump_mask = (
        (kmd.forest.doftype == int(NodeType.jump))
        & torch.all(
            kf_network.dof_mask[
                :, : int(JumpDOFTypes.RBdel_gamma) + 1
            ],
            dim=1,
        )
    )
    jump_node_list = torch.nonzero(jump_mask, as_tuple=False).flatten()
    if jump_node_list.numel() != pose_stack.n_poses:
        raise ValueError("expected exactly one movable ligand jump per pose")
    jump_pose = torch.div(
        kmd.forest.id[jump_node_list],
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
    ligand_nodes = _ligand_kinforest_nodes(
        pose_stack,
        docking_dofs.ligand_block,
        block_type.n_atoms,
        kmd.forest.id,
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
