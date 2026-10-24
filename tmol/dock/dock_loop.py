"""A minimal batched score--minimize--select docking generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import attr
import torch

from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    build_ligand_docking_dofs,
)
from tmol.dock.scoring_backend import MinimizationResult, ScoringBackend
from tmol.kinematics.compiled import inverse_kin
from tmol.kinematics import BondDOFTypes, JumpDOFTypes, NodeType
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.pose import PoseStack
from tmol.pose import repeat_pose_stack


@dataclass(frozen=True)
class DockingPerturbation:
    """Uniform half-widths for the initial ligand DOF jitter.

    The defaults sample each translation DOF uniformly within +/-0.25 A,
    each jump-rotation DOF within +/-5 degrees, and each ligand torsion raw
    DOF within +/-10 degrees. Rotation and torsion values are stored in
    radians because that is the kinematics representation.
    """

    translation_angstrom: float = 0.25
    rotation_radians: float = math.radians(5.0)
    torsion_radians: float = math.radians(10.0)

    def __post_init__(self) -> None:
        for name, value in (
            ("translation_angstrom", self.translation_angstrom),
            ("rotation_radians", self.rotation_radians),
            ("torsion_radians", self.torsion_radians),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class DockGenerationResult:
    """Outputs of one batched docking generation."""

    minimized_pose_stack: PoseStack
    initial_energies: torch.Tensor
    final_energies: torch.Tensor
    selection: torch.Tensor
    best_index: torch.Tensor


def _inverse_dofs(
    pose_stack: PoseStack, kin_module: PoseStackKinematicsModule
) -> torch.Tensor:
    """Convert PoseStack-ordered coordinates to raw kinematic DOFs."""

    kmd = kin_module.kmd
    dtype = pose_stack.coords.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("docking requires float32 or float64 coordinates")
    kin_coords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=dtype, device=pose_stack.device
    )
    kin_coords[1:] = pose_stack.coords.reshape(-1, 3)[
        kmd.forest.id[1:].to(torch.int64)
    ]
    return inverse_kin(
        kin_coords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )


def _perturb_ligand_dofs(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    perturb: DockingPerturbation,
    seed: int,
) -> PoseStack:
    """Jitter free ligand raw DOFs in poses 1..N-1 and rebuild coordinates."""

    kin_module = PoseStackKinematicsModule(
        pose_stack, docking_dofs.fold_forest
    )
    kmd = kin_module.kmd
    raw_dofs = _inverse_dofs(pose_stack, kin_module)
    dof_mask = MinimizerMap(
        pose_stack, kmd, docking_dofs.move_map
    ).dof_mask
    nodes, dof_columns = torch.nonzero(dof_mask, as_tuple=True)

    target_atoms = kmd.forest.id[nodes].to(torch.int64)
    pose_for_dof = torch.div(
        target_atoms, pose_stack.max_n_pose_atoms, rounding_mode="floor"
    )
    jittered = pose_for_dof > 0
    nodes = nodes[jittered]
    dof_columns = dof_columns[jittered]

    node_types = kmd.forest.doftype[nodes]
    jump = node_types == int(NodeType.jump)
    translation = jump & (dof_columns <= int(JumpDOFTypes.RBz))
    rotation = jump & (
        (dof_columns >= int(JumpDOFTypes.RBdel_alpha))
        & (dof_columns <= int(JumpDOFTypes.RBdel_gamma))
    )
    torsion = (node_types == int(NodeType.bond)) & (
        (dof_columns == int(BondDOFTypes.phi_p))
        | (dof_columns == int(BondDOFTypes.phi_c))
    )
    if not bool(torch.all(translation | rotation | torsion)):
        raise ValueError("ligand docking MoveMap contains an unsupported free DOF")

    scales = torch.empty(
        nodes.shape, dtype=raw_dofs.dtype, device=pose_stack.device
    )
    scales[translation] = perturb.translation_angstrom
    scales[rotation] = perturb.rotation_radians
    scales[torsion] = perturb.torsion_radians

    generator = torch.Generator(device=pose_stack.device)
    generator.manual_seed(seed)
    unit_jitter = 2.0 * torch.rand(
        nodes.shape,
        dtype=raw_dofs.dtype,
        device=pose_stack.device,
        generator=generator,
    ) - 1.0
    perturbed_dofs = raw_dofs.clone()
    perturbed_dofs[nodes, dof_columns] += unit_jitter * scales
    kin_coords = kin_module(perturbed_dofs)

    # Forward kinematics rebuilds the whole forest. Copy only the ligand atoms
    # for perturbed poses so receptor coordinates and pose 0 remain bitwise
    # identical to the stacked reference before minimization.
    flat_coords = pose_stack.coords.clone().reshape(-1, 3)
    all_target_atoms = kmd.forest.id[1:].to(torch.int64)
    pose_for_atom = torch.div(
        all_target_atoms,
        pose_stack.max_n_pose_atoms,
        rounding_mode="floor",
    )
    atom_in_pose = torch.remainder(
        all_target_atoms, pose_stack.max_n_pose_atoms
    )
    ligand_offsets = pose_stack.block_coord_offset64[
        pose_for_atom, docking_dofs.ligand_block
    ]
    block_type_index = int(
        pose_stack.block_type_ind64[0, docking_dofs.ligand_block]
    )
    ligand_n_atoms = pose_stack.packed_block_types.active_block_types[
        block_type_index
    ].n_atoms
    copy_atom = (
        (pose_for_atom > 0)
        & (atom_in_pose >= ligand_offsets)
        & (atom_in_pose < ligand_offsets + ligand_n_atoms)
    )
    flat_coords[all_target_atoms[copy_atom]] = kin_coords[1:][copy_atom].to(
        flat_coords.dtype
    )
    return attr.evolve(pose_stack, coords=flat_coords.view_as(pose_stack.coords))


def run_one_generation(
    pose_stack: PoseStack,
    ligand_block: int,
    backend: ScoringBackend,
    *,
    n_pop: int,
    seed: int,
    perturb: DockingPerturbation,
) -> DockGenerationResult:
    """Run one continuous docking generation over a batched population.

    The input must contain one reference protein--ligand pose. The reference is
    replicated ``n_pop`` times, pose 0 is kept unperturbed, and poses 1 onward
    receive the uniform ligand-only jitter described by ``perturb``. The full
    population is scored, minimized, rescored, and sorted without per-pose
    Python scoring or minimization loops.
    """

    if pose_stack.n_poses != 1:
        raise ValueError("pose_stack must contain exactly one reference pose")
    if not isinstance(n_pop, int) or isinstance(n_pop, bool) or n_pop < 1:
        raise ValueError("n_pop must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(perturb, DockingPerturbation):
        raise TypeError("perturb must be a DockingPerturbation")

    population = repeat_pose_stack(pose_stack, n_pop)

    docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    population = _perturb_ligand_dofs(
        population, docking_dofs, perturb, seed
    )
    initial_energies = backend.score(population)
    expected_shape = (n_pop,)
    if initial_energies.shape != expected_shape:
        raise ValueError(
            f"backend.score returned {tuple(initial_energies.shape)} before "
            f"minimization; expected {expected_shape}"
        )

    minimize_result = backend.minimize(population, docking_dofs)
    minimized = (
        minimize_result.pose_stack
        if isinstance(minimize_result, MinimizationResult)
        else minimize_result
    )
    if minimized.n_poses != n_pop:
        raise ValueError(
            f"backend.minimize returned {minimized.n_poses} poses; expected {n_pop}"
        )

    final_energies = backend.score(minimized)
    if final_energies.shape != expected_shape:
        raise ValueError(
            f"backend.score returned {tuple(final_energies.shape)} after "
            f"minimization; expected {expected_shape}"
        )

    selection = torch.argsort(final_energies)
    return DockGenerationResult(
        minimized_pose_stack=minimized,
        initial_energies=initial_energies,
        final_energies=final_energies,
        selection=selection,
        best_index=torch.argmin(final_energies),
    )
