"""Vectorized GALigandDock genetic operators for one ligand population.

The control flow mirrors Rosetta ``GAOptimizer.cc:104-113``: generate an
offspring batch, optimize the whole batch, then merge it into the parent pool.
The gene is exactly the MoveMap-freed raw DOFs: the ligand jump's three
translations and three delta rotations followed by logical ligand chis and
then selected receptor-sidechain chis. Logical chis use the docking-local
sign/offset mapping in ``ligand_dofs.py``.
Genes are only converted to coordinates by forward kinematics; minimized genes
come directly from the optimizer's freed DOFs, never from inverse kinematics of
the minimized pose. Ring genes are out of scope.
Continuous scoring has no grid smoothing or fa_rep ramp, so this module runs
the same schedule un-ramped until a grid backend implements those knobs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import attr
import numpy
import torch

from tmol.dock.conformer_seed import generate_conformer_library
from tmol.dock._diversity import ligand_heavy_atom_rms, select_diverse_low_energy
from tmol.dock.ligand_aligner import LigandAligner
from tmol.dock.ligand_dofs import (
    GeneIndex,
    LigandDockingDOFs,
    _inverse_dofs,
    _sidechain_coordinate_metadata,
    _sidechain_logical_torsions_from_coords,
    _wrap_angle,
    build_ligand_docking_dofs,
)
from tmol.dock.scoring_backend import (
    GridBackend,
    MinimizationResult,
    ScoringBackend,
)
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.pose import PoseStack
from tmol.pose import repeat_pose_stack
from tmol.score.gridscore._movable import (
    has_nonpolymer_connection,
    resolve_ligand_block,
)

if TYPE_CHECKING:
    from tmol.dock.finalization import DockingFinalization


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class GASchedule:
    """Single-stage VSX schedule for the continuous backend."""

    generations: int = 5
    npool: int = 50
    rmsdthreshold: float = 2.0
    pmut: float = 0.2
    maxiter: int = 50
    initial_perturbation: DockingPerturbation = DockingPerturbation()


@dataclass(frozen=True)
class RandomPlacement:
    """Ranges for an unbiased initial ligand population.

    The ligand heavy-atom centroid is placed uniformly inside the supplied
    pocket axis-aligned bounding box.  When no box is supplied, the reference
    ligand-heavy-atom box is expanded by ``pocket_padding_angstrom`` on every
    side.  Orientations are uniform on SO(3), and each logical torsion is
    uniform on ``(-pi, pi]``.
    """

    pocket_padding_angstrom: float = 2.0

    def __post_init__(self) -> None:
        pass


_GENE_INDEX_FIELDS = (
    "jump_nodes",
    "torsion_nodes",
    "torsion_dof_types",
    "torsion_signs",
    "torsion_offsets",
    "ligand_nodes",
)


@dataclass(frozen=True)
class LigandGeneLayout:
    """Map ``[pose, jump | ligand chis | receptor chis]`` to coordinates."""

    template: PoseStack
    docking_dofs: LigandDockingDOFs
    kin_module: PoseStackKinematicsModule
    base_raw_dofs: torch.Tensor
    index: GeneIndex
    ligand_local_atoms: torch.Tensor
    heavy_local_atoms: torch.Tensor
    sidechain_nodes: torch.Tensor
    sidechain_atom_indices: torch.Tensor
    sidechain_atom_influence: torch.Tensor

    @classmethod
    def from_pose_stack(
        cls, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> "LigandGeneLayout":
        kin_module = PoseStackKinematicsModule(pose_stack, docking_dofs.fold_forest)
        kmd = kin_module.kmd
        raw_dofs = _inverse_dofs(pose_stack, kin_module)
        minmap = MinimizerMap(pose_stack, kmd, docking_dofs.move_map)

        block_type_index = int(
            pose_stack.block_type_ind64[0, docking_dofs.ligand_block]
        )
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        index = GeneIndex.build(
            pose_stack, docking_dofs, kmd.forest, minmap.dof_mask, raw_dofs.dtype
        )
        heavy_local_atoms = torch.nonzero(
            ~pose_stack.packed_block_types.atom_is_hydrogen[
                block_type_index, : block_type.n_atoms
            ].to(torch.bool),
            as_tuple=False,
        ).flatten()
        (
            sidechain_nodes,
            sidechain_atom_indices,
            sidechain_atom_influence,
        ) = _sidechain_coordinate_metadata(pose_stack, docking_dofs, kmd.forest)

        return cls(
            template=pose_stack,
            docking_dofs=docking_dofs,
            kin_module=kin_module,
            base_raw_dofs=raw_dofs,
            index=index,
            ligand_local_atoms=torch.arange(
                block_type.n_atoms,
                dtype=torch.int64,
                device=pose_stack.device,
            ),
            heavy_local_atoms=heavy_local_atoms,
            sidechain_nodes=sidechain_nodes,
            sidechain_atom_indices=sidechain_atom_indices,
            sidechain_atom_influence=sidechain_atom_influence,
        )

    def __getattr__(self, name: str):
        if name in _GENE_INDEX_FIELDS:
            return getattr(self.index, name)
        raise AttributeError(name)

    @property
    def n_dof(self) -> int:
        return 6 + self.index.torsion_nodes.shape[1]

    def genes_from_raw_dofs(self, raw_dofs: torch.Tensor) -> torch.Tensor:
        """Gather the fixed-order gene from a raw kinematic DOF tensor."""

        if raw_dofs.shape != self.base_raw_dofs.shape:
            raise ValueError(
                "raw DOF shape does not match the gene layout: "
                f"{tuple(raw_dofs.shape)} != {tuple(self.base_raw_dofs.shape)}"
            )
        return self.index.genes_from_raw_dofs(raw_dofs)

    def genes_from_pose_stack(self, pose_stack: PoseStack) -> torch.Tensor:
        """Extract the starting template gene from its freed raw DOFs.

        This inverse-kinematics path exists only to seed the initial pool. The
        generation loop receives minimized genes directly from the optimizer.
        """

        if pose_stack.n_poses != self.template.n_poses:
            raise ValueError("pose stack size does not match the gene layout")
        raw_dofs = _inverse_dofs(pose_stack, self.kin_module)
        genes = self.genes_from_raw_dofs(raw_dofs)
        n_sidechain_torsions = len(self.docking_dofs.sidechain_torsions)
        if n_sidechain_torsions:
            # Recover each sidechain gene from its pose-local named chi delta.
            # Starting from the base raw-DOF gene preserves the native path's
            # bit identity while avoiding the cross-pose corruption observed
            # in batched inverse-kinematics sidechain columns.
            base_genes = self.genes_from_raw_dofs(self.base_raw_dofs)[
                :, -n_sidechain_torsions:
            ]
            base_chis = _sidechain_logical_torsions_from_coords(
                self.template, self.docking_dofs
            )
            chis = _sidechain_logical_torsions_from_coords(
                pose_stack, self.docking_dofs
            )
            sidechain_genes = base_genes + _wrap_angle(chis - base_chis)
            in_canonical_range = (sidechain_genes > -math.pi) & (
                sidechain_genes <= math.pi
            )
            genes[:, -n_sidechain_torsions:] = torch.where(
                in_canonical_range,
                sidechain_genes,
                _wrap_angle(sidechain_genes),
            )
        return genes

    def raw_dofs_from_genes(self, genes: torch.Tensor) -> torch.Tensor:
        """Write genes into a clone of the template raw-DOF tensor."""

        expected = (self.template.n_poses, self.n_dof)
        if genes.shape != expected:
            raise ValueError(
                f"genes have shape {tuple(genes.shape)}; expected {expected}"
            )

        raw_dofs = self.base_raw_dofs.clone()
        self.index.write_raw_dofs(raw_dofs, genes)
        return raw_dofs

    def pose_stack_from_genes(self, genes: torch.Tensor) -> PoseStack:
        """Rebuild ligand and movable-sidechain coordinates from genes."""

        raw_dofs = self.raw_dofs_from_genes(genes)
        kin_coords = self.kin_module(raw_dofs)
        coords = self.template.coords.clone()
        pose_indices = torch.arange(
            self.template.n_poses,
            dtype=torch.int64,
            device=self.template.device,
        )
        atom_indices = (
            self.template.block_coord_offset64[:, self.docking_dofs.ligand_block, None]
            + self.ligand_local_atoms[None, :]
        )
        coords[pose_indices[:, None], atom_indices] = kin_coords[self.ligand_nodes].to(
            coords.dtype
        )
        if self.sidechain_nodes.shape[1]:
            rebuilt_sidechains = kin_coords[self.sidechain_nodes].to(coords.dtype)
            coords[pose_indices[:, None], self.sidechain_atom_indices] = (
                rebuilt_sidechains
            )
        return attr.evolve(self.template, coords=coords)

    def ligand_heavy_coords(self, pose_stack: PoseStack) -> torch.Tensor:
        """Gather ligand heavy atoms as ``[pose, heavy_atom, xyz]``."""

        return _block_heavy_coords(
            pose_stack,
            self.docking_dofs.ligand_block,
            self.heavy_local_atoms,
        )


def _block_heavy_coords(
    pose_stack: PoseStack,
    block: int,
    heavy_local_atoms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather one homogeneous block's heavy-atom coordinates."""

    block_type_indices = pose_stack.block_type_ind64[:, block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError(f"block {block} is absent from at least one pose")
    if bool(torch.any(block_type_indices != block_type_indices[0])):
        raise ValueError(f"all poses must use the same type at block {block}")
    block_type_index = int(block_type_indices[0])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    if heavy_local_atoms is None:
        heavy_local_atoms = torch.nonzero(
            ~pose_stack.packed_block_types.atom_is_hydrogen[
                block_type_index, : block_type.n_atoms
            ].to(torch.bool),
            as_tuple=False,
        ).flatten()
    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    atom_indices = (
        pose_stack.block_coord_offset64[:, block, None] + heavy_local_atoms[None, :]
    )
    return pose_stack.coords[pose_indices[:, None], atom_indices]


def select_pocket_residues(
    pose_stack: PoseStack,
    ligand: str | int,
    cutoff: float = 6.0,
) -> torch.Tensor:
    """Return sorted polymer residues (block indices) contacting ligand heavy atoms.

    A block is selected when any of its heavy atoms is within ``cutoff`` of a
    ligand heavy atom in any pose. Non-polymer blocks, including cofactors and
    solvent, are excluded. Polymer residues carrying a non-polymer covalent
    connection (e.g. a disulfide-bonded cysteine) are also excluded, because
    flexible-sidechain kinematics cannot preserve that bond; passing such a
    residue explicitly is rejected downstream by ``validated_sidechain_blocks``.
    Chi-less residues (e.g. ALA, GLY) are excluded as well: they have nothing to
    move, so the returned set is directly usable as flexible ``sidechain_blocks``.
    """
    ligand_block = resolve_ligand_block(pose_stack, ligand)
    cutoff = float(cutoff)

    ligand_heavy = _block_heavy_coords(pose_stack, ligand_block)
    selected = []
    for block in range(pose_stack.max_n_blocks):
        if block == ligand_block:
            continue
        block_type_indices = pose_stack.block_type_ind64[:, block]
        if bool(torch.any(block_type_indices < 0)):
            continue
        if bool(torch.any(block_type_indices != block_type_indices[0])):
            raise ValueError(f"all poses must use the same type at block {block}")
        block_type = pose_stack.packed_block_types.active_block_types[
            int(block_type_indices[0])
        ]
        if not block_type.properties.polymer.is_polymer:
            continue
        if has_nonpolymer_connection(pose_stack, block, int(block_type_indices[0])):
            continue
        if int(block_type.n_sc_torsions) == 0:
            continue
        receptor_heavy = _block_heavy_coords(pose_stack, block)
        if bool(torch.any(torch.cdist(receptor_heavy, ligand_heavy) <= cutoff)):
            selected.append(block)
    return torch.tensor(selected, dtype=torch.int64, device=pose_stack.device)


@dataclass(frozen=True)
class GAPool:
    genes: torch.Tensor
    pose_stack: PoseStack
    energies: torch.Tensor
    n_diverse: int
    selected_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class GAResult:
    pool: GAPool
    best_energy_history: torch.Tensor
    diverse_count_history: tuple[int, ...]
    ranked_genes: torch.Tensor | None = None
    ranked_energies: torch.Tensor | None = None
    best_genes: torch.Tensor | None = None
    best_energies: torch.Tensor | None = None
    best_pose_stack: PoseStack | None = None


@dataclass(frozen=True)
class DockingResult:
    """Ranked end-to-end docking output for one input ligand."""

    best_pose_stack: PoseStack
    best_energies: torch.Tensor
    best_genes: torch.Tensor
    ranked_energies: torch.Tensor
    ranked_genes: torch.Tensor
    ga_result: GAResult
    finalization: DockingFinalization | None = None

    # Finalization outputs, exposed here as ``None`` until finalization has run.
    _FINALIZATION_FIELDS = {
        "finalized_pose_stack": "pose_stack",
        "pre_refine_energies": "pre_refine_energies",
        "complex_energies": "complex_energies",
        "free_ligand_energies": "free_ligand_energies",
        "free_receptor_energies": "free_receptor_energies",
        "nchi": "nchi",
        "dH": "dH",
        "TdS": "TdS",
        "dG": "dG",
    }

    def __getattr__(self, name: str):
        field = DockingResult._FINALIZATION_FIELDS.get(name)
        if field is None:
            raise AttributeError(name)
        return None if self.finalization is None else getattr(self.finalization, field)


def _place_ligand_in_reference(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    ligand_pose_stack: PoseStack,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    compact_sidechain_blocks: tuple[int, ...] = (),
    sidechain_atom_indices: torch.Tensor | None = None,
) -> PoseStack:
    """Restore ligand and moved sidechain atoms into a reference complex."""

    if ligand_pose_stack.n_poses != reference_pose_stack.n_poses:
        raise ValueError("ligand and reference pose counts must match")
    block_type_indices = reference_pose_stack.block_type_ind64[:, ligand_block]
    if not bool(
        torch.all(block_type_indices == ligand_pose_stack.block_type_ind64[:, 0])
    ):
        raise ValueError("ligand and reference block types must match")
    block_type_index = int(block_type_indices[0])
    n_atoms = reference_pose_stack.packed_block_types.active_block_types[
        block_type_index
    ].n_atoms
    if not sidechain_blocks and ligand_pose_stack.coords.shape[1] != n_atoms:
        raise ValueError("ligand atom counts must match")

    pose_indices = torch.arange(
        reference_pose_stack.n_poses,
        dtype=torch.int64,
        device=reference_pose_stack.device,
    )
    atom_indices = (
        reference_pose_stack.block_coord_offset64[:, ligand_block, None]
        + torch.arange(n_atoms, dtype=torch.int64, device=reference_pose_stack.device)[
            None, :
        ]
    )
    coords = reference_pose_stack.coords.clone()
    coords[pose_indices[:, None], atom_indices] = ligand_pose_stack.coords[:, :n_atoms]
    if sidechain_blocks:
        if sidechain_atom_indices is None or sidechain_atom_indices.ndim != 1:
            raise ValueError(
                "flexible restoration requires one-dimensional sidechain atom indices"
            )
        if len(compact_sidechain_blocks) != len(sidechain_blocks):
            raise ValueError(
                "flexible restoration requires compact sidechain block mapping"
            )
        if ligand_pose_stack.max_n_blocks < 1 + len(sidechain_blocks):
            raise ValueError("movable pose is missing requested sidechain blocks")
        for compact_block, source_block in zip(
            compact_sidechain_blocks, sidechain_blocks
        ):
            compact_start = int(
                ligand_pose_stack.block_coord_offset64[0, compact_block]
            )
            compact_type = ligand_pose_stack.block_type(0, compact_block)
            compact_end = compact_start + compact_type.n_atoms
            selected = sidechain_atom_indices[
                (sidechain_atom_indices >= compact_start)
                & (sidechain_atom_indices < compact_end)
            ]
            if selected.numel() == 0:
                continue
            local_atoms = selected - compact_start
            source_type_indices = reference_pose_stack.block_type_ind64[:, source_block]
            if not bool(
                torch.all(
                    source_type_indices
                    == ligand_pose_stack.block_type_ind64[:, compact_block]
                )
            ):
                raise ValueError("sidechain and reference block types must match")
            target_atoms = (
                reference_pose_stack.block_coord_offset64[:, source_block, None]
                + local_atoms[None, :]
            )
            coords[pose_indices[:, None], target_atoms] = ligand_pose_stack.coords[
                pose_indices[:, None], selected[None, :]
            ]
    return attr.evolve(reference_pose_stack, coords=coords)


def _unit_vectors(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    vectors = torch.randn((count, 3), dtype=dtype, device=device, generator=generator)
    return vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(
        torch.finfo(dtype).tiny
    )


def _canonical_torsions(torsions: torch.Tensor) -> torch.Tensor:
    """Wrap torsions to the unique interval ``(-pi, pi]``."""

    wrapped = torch.remainder(torsions + math.pi, 2 * math.pi) - math.pi
    return torch.where(
        wrapped == -math.pi,
        torch.full_like(wrapped, math.pi),
        wrapped,
    )


def _canonicalize_genes(genes: torch.Tensor) -> torch.Tensor:
    """Wrap only logical torsions; preserve all six freed jump DOFs."""
    return torch.cat(
        (genes[:, :6], _canonical_torsions(genes[:, 6:])),
        dim=1,
    )


def _zyx_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Build tmol's row-vector ZYX rotation matrix from ``[..., 3]`` angles."""

    alpha, beta, gamma = torch.unbind(euler, dim=-1)
    si, sj, sk = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)
    ci, cj, ck = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    return torch.stack(
        (
            cj * ck,
            cj * sk,
            -sj,
            sj * si * ck - ci * sk,
            sj * si * sk + ci * ck,
            cj * si,
            sj * ci * ck + si * sk,
            sj * ci * sk - si * ck,
            cj * ci,
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))


def _rotation_matrix_to_zyx(rotation: torch.Tensor) -> torch.Tensor:
    """Invert ``_zyx_rotation_matrix`` with common.hh's canonical branch."""

    cy = torch.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 0, 1] ** 2)
    singular = cy < 1.0e-6
    alpha = torch.where(
        singular,
        torch.atan2(-rotation[..., 2, 1], rotation[..., 1, 1]),
        torch.atan2(rotation[..., 1, 2], rotation[..., 2, 2]),
    )
    beta = torch.atan2(-rotation[..., 0, 2], cy)
    gamma = torch.where(
        singular,
        torch.zeros_like(cy),
        torch.atan2(rotation[..., 0, 1], rotation[..., 0, 0]),
    )
    return torch.stack((alpha, beta, gamma), dim=-1)


def _quaternion_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Build Rosetta's column-vector rotation matrix from ``[..., w,x,y,z]``."""

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _apply_rosetta_rotation_mutation(
    parent_euler: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Left-compose Rosetta's ``(cos(a), sin(a)*axis)`` perturbation.

    Rosetta rotation matrices act on column vectors, while tmol's kinematics
    stores their transposes for row-vector homogeneous transforms. Composition
    is therefore performed in Rosetta's convention and transposed only at the
    ZYX conversion boundaries.
    """

    perturbation_quaternion = torch.cat(
        (torch.cos(angle)[..., None], torch.sin(angle)[..., None] * axis),
        dim=-1,
    )
    parent_rotation = _zyx_rotation_matrix(parent_euler).transpose(-1, -2)
    perturbation_rotation = _quaternion_rotation_matrix(perturbation_quaternion)
    composed_rotation = perturbation_rotation @ parent_rotation
    return _rotation_matrix_to_zyx(composed_rotation.transpose(-1, -2))


def mutate(
    genes: torch.Tensor,
    *,
    generator: torch.Generator,
    rtmutation_rate: float = 0.5,
    torsmutation_rate: float = 0.1,
    translation_width: float = 2.0,
    rotation_width: float = math.radians(60.0),
    torsion_width: float = math.radians(120.0),
) -> torch.Tensor:
    """Mutate a gene batch with Rosetta's rigid-body-or-chi semantics.

    ``LigandConformer.cc:765-820`` selects rigid-body mutation with probability
    0.5; otherwise each chi mutates with probability 0.1 and one chi is forced
    when none was selected. Translation has a uniform radius and chi jitter is
    uniform in its width. Rotation draws Rosetta's Gaussian quaternion
    parameter about a random axis and left-composes the perturbation with the
    parent rotation. The composed orientation is written back as raw ZYX
    delta-Euler DOFs for forward kinematics.
    """
    n_genes, n_dof = genes.shape
    device, dtype = genes.device, genes.dtype
    randomize_rt = (
        torch.rand((n_genes,), device=device, generator=generator) < rtmutation_rate
    )
    result = genes.clone()

    translation_axis = _unit_vectors(
        n_genes, dtype=dtype, device=device, generator=generator
    )
    translation_length = (
        torch.rand((n_genes, 1), dtype=dtype, device=device, generator=generator)
        * translation_width
    )
    result[:, :3] += randomize_rt[:, None] * translation_axis * translation_length

    if rotation_width > math.pi:
        random_values = torch.rand(
            (n_genes, 3), dtype=dtype, device=device, generator=generator
        )
        u1 = random_values[:, 0]
        u2 = 2 * math.pi * random_values[:, 1]
        u3 = 2 * math.pi * random_values[:, 2]
        random_quaternion = torch.stack(
            (
                torch.sqrt(1 - u1) * torch.sin(u2),
                torch.sqrt(1 - u1) * torch.cos(u2),
                torch.sqrt(u1) * torch.sin(u3),
                torch.sqrt(u1) * torch.cos(u3),
            ),
            dim=1,
        )
        random_rotation = _quaternion_rotation_matrix(random_quaternion)
        mutated_rotation = _rotation_matrix_to_zyx(random_rotation.transpose(-1, -2))
    else:
        rotation_axis = _unit_vectors(
            n_genes, dtype=dtype, device=device, generator=generator
        )
        angle = rotation_width * torch.randn(
            (n_genes,), dtype=dtype, device=device, generator=generator
        )
        mutated_rotation = _apply_rosetta_rotation_mutation(
            genes[:, 3:6], rotation_axis, angle
        )
    result[:, 3:6] = torch.where(randomize_rt[:, None], mutated_rotation, genes[:, 3:6])

    n_torsions = n_dof - 6
    if n_torsions:
        mutate_torsion = (
            torch.rand((n_genes, n_torsions), device=device, generator=generator)
            < torsmutation_rate
        ) & ~randomize_rt[:, None]
        force = ~randomize_rt & ~torch.any(mutate_torsion, dim=1)
        forced_index = torch.randint(
            n_torsions, (n_genes, 1), device=device, generator=generator
        )
        forced = torch.zeros_like(mutate_torsion)
        forced.scatter_(1, forced_index, force[:, None])
        mutate_torsion |= forced

        if torsion_width > math.pi:
            mutated_torsions = (
                2
                * math.pi
                * torch.rand(
                    (n_genes, n_torsions),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
            )
        else:
            delta = torsion_width * (
                1
                - 2
                * torch.rand(
                    (n_genes, n_torsions),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
            )
            mutated_torsions = genes[:, 6:] + delta
        result[:, 6:] = torch.where(mutate_torsion, mutated_torsions, genes[:, 6:])
    return _canonicalize_genes(result)


def crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniform crossover with rigid body inherited as one indivisible unit.

    Rosetta selects the complete rigid body from one parent, then each chi
    independently, forcing one partner-2 chi when all independent choices took
    parent 1 (``LigandConformer.cc:862-897``).
    """
    if parent1.shape[1] < 6:
        raise ValueError("crossover genes need a six-value rigid body")
    n_genes, n_dof = parent1.shape
    take_rb_1 = (
        torch.rand((n_genes, 1), device=parent1.device, generator=generator) < 0.5
    )
    result = torch.empty_like(parent1)
    result[:, :6] = torch.where(take_rb_1, parent1[:, :6], parent2[:, :6])

    n_torsions = n_dof - 6
    if n_torsions:
        take_chi_1 = (
            torch.rand(
                (n_genes, n_torsions),
                device=parent1.device,
                generator=generator,
            )
            < 0.5
        )
        force_partner2 = torch.all(take_chi_1, dim=1)
        forced_index = torch.randint(
            n_torsions,
            (n_genes, 1),
            device=parent1.device,
            generator=generator,
        )
        forced_partner2 = torch.zeros_like(take_chi_1)
        forced_partner2.scatter_(1, forced_index, force_partner2[:, None])
        take_chi_1 &= ~forced_partner2
        result[:, 6:] = torch.where(take_chi_1, parent1[:, 6:], parent2[:, 6:])
    return _canonicalize_genes(result)


def next_generation(
    genes: torch.Tensor,
    npoolout: int,
    pmut: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create offspring by cyclic parent assignment and mutation/crossover.

    This is the batched equivalent of ``GAOptimizer.cc:220-258``. Parent ``i``
    cycles modulo the input pool; mutation is chosen with ``pmut`` and a
    crossover partner is sampled uniformly from every other parent.
    """
    npoolin = genes.shape[0]
    parent_indices = (
        torch.arange(npoolout, dtype=torch.int64, device=genes.device) % npoolin
    )
    parents = genes[parent_indices]
    mutate_choice = (
        torch.rand((npoolout,), device=genes.device, generator=generator) <= pmut
    )
    if npoolin == 1:
        mutate_choice = torch.ones_like(mutate_choice)
        partners = parents
    else:
        partner_indices = torch.randint(
            npoolin - 1,
            (npoolout,),
            device=genes.device,
            generator=generator,
        )
        partner_indices += (partner_indices >= parent_indices).to(torch.int64)
        partners = genes[partner_indices]
    mutated = mutate(parents, generator=generator)
    crossed = crossover(parents, partners, generator=generator)
    return torch.where(mutate_choice[:, None], mutated, crossed)


def optimize_generation(
    genes: torch.Tensor,
    layout: LigandGeneLayout,
    backend: ScoringBackend,
    *,
    maxiter: int,
) -> GAPool:
    """Score and locally minimize one whole population as a batch.

    This mirrors the score-then-optimize operation at
    ``GAOptimizer.cc:195-208`` while delegating the batched objective to the
    Phase 2 backend.
    """

    proposed = layout.pose_stack_from_genes(genes)
    proposed_raw_dofs = layout.raw_dofs_from_genes(genes)
    premin = backend.score(proposed)
    expected = (genes.shape[0],)
    if premin.shape != expected:
        raise ValueError(
            f"backend.score returned {tuple(premin.shape)}; expected {expected}"
        )
    minimize_result = backend.minimize(
        proposed,
        layout.docking_dofs,
        maxiter=maxiter,
        initial_raw_dofs=proposed_raw_dofs,
    )
    if not isinstance(minimize_result, MinimizationResult):
        raise TypeError(
            "GA backends must return MinimizationResult with optimizer genes"
        )
    minimized = minimize_result.pose_stack
    minimized_genes = minimize_result.genes
    if minimized_genes is None:
        raise ValueError("GA minimization did not return a freed-DOF gene")
    if minimized_genes.shape != genes.shape:
        raise ValueError(
            "backend.minimize returned genes with shape "
            f"{tuple(minimized_genes.shape)}; expected {tuple(genes.shape)}"
        )
    if minimized_genes.device != genes.device:
        raise ValueError("minimized genes moved to a different device")
    if minimized_genes.dtype != genes.dtype:
        raise ValueError("minimized genes changed dtype")
    energies = backend.score(minimized)
    if energies.shape != expected:
        raise ValueError(
            f"backend.score returned {tuple(energies.shape)}; expected {expected}"
        )
    return GAPool(
        genes=minimized_genes.detach(),
        pose_stack=minimized,
        energies=energies.detach(),
        n_diverse=genes.shape[0],
    )


def update_pool(
    parents: GAPool,
    offspring: GAPool,
    layout: LigandGeneLayout,
    *,
    npool: int,
    rmsdthreshold: float,
) -> GAPool:
    """Greedily retain energy-ranked RMS-diverse genes, then fill if needed.

    Rosetta concatenates offspring and parents, sorts by energy, greedily keeps
    candidates whose heavy-atom RMS is not below the cutoff, and finally fills
    any missing slots with rejected candidates (``GAOptimizer.cc:268-313``).
    Consequently ``n_diverse < npool`` documents a filled pool that cannot
    satisfy the cutoff, matching Rosetta rather than returning a short pool.
    """

    genes = torch.cat((offspring.genes, parents.genes), dim=0)
    energies = torch.cat((offspring.energies, parents.energies), dim=0)
    # RMS niching is defined by the genes themselves. Rebuild both populations
    # with forward kinematics so no coordinate-to-gene inversion participates
    # in the distance calculation.
    offspring_from_genes = layout.pose_stack_from_genes(offspring.genes)
    parents_from_genes = layout.pose_stack_from_genes(parents.genes)
    coords = torch.cat(
        (
            layout.ligand_heavy_coords(offspring_from_genes),
            layout.ligand_heavy_coords(parents_from_genes),
        ),
        dim=0,
    )
    if genes.shape[0] <= npool:
        raise ValueError("combined parent and offspring pools must exceed npool")
    selected_tensor, n_diverse = select_diverse_low_energy(
        energies, coords, npool=npool, rmsdthreshold=rmsdthreshold
    )
    n_diverse = int(n_diverse)

    candidate_coords = torch.cat(
        (offspring.pose_stack.coords, parents.pose_stack.coords), dim=0
    )
    selected_pose_stack = attr.evolve(
        parents.pose_stack, coords=candidate_coords[selected_tensor]
    )
    return GAPool(
        genes=genes[selected_tensor],
        pose_stack=selected_pose_stack,
        energies=energies[selected_tensor],
        n_diverse=n_diverse,
        selected_indices=selected_tensor,
    )


def _initial_genes(
    layout: LigandGeneLayout,
    perturbation: DockingPerturbation,
    generator: torch.Generator,
) -> torch.Tensor:
    genes = layout.genes_from_pose_stack(layout.template)
    unit_jitter = (
        2
        * torch.rand(
            genes.shape,
            dtype=genes.dtype,
            device=genes.device,
            generator=generator,
        )
        - 1
    )
    jitter_rows = torch.arange(genes.shape[0], device=genes.device) > 0
    result = genes.clone()
    result[:, :3] += (
        jitter_rows[:, None] * unit_jitter[:, :3] * perturbation.translation_angstrom
    )
    result[:, 3:6] += (
        jitter_rows[:, None] * unit_jitter[:, 3:6] * perturbation.rotation_radians
    )
    result[:, 6:] += (
        jitter_rows[:, None] * unit_jitter[:, 6:] * perturbation.torsion_radians
    )
    return _canonicalize_genes(result)


def _normalize_pocket_bounds(
    layout: LigandGeneLayout,
    *,
    pocket_bounds: torch.Tensor | None,
    placement: RandomPlacement,
) -> torch.Tensor:
    """Return the reference ligand's pocket box as ``[min/max, xyz]``."""

    if pocket_bounds is None:
        heavy = layout.ligand_heavy_coords(layout.template)[0]
        padding = placement.pocket_padding_angstrom
        return torch.stack(
            (
                torch.amin(heavy, dim=0) - padding,
                torch.amax(heavy, dim=0) + padding,
            ),
            dim=0,
        )
    bounds = torch.as_tensor(
        pocket_bounds, dtype=layout.base_raw_dofs.dtype, device=layout.template.device
    )
    return bounds


def _random_initial_genes(
    layout: LigandGeneLayout,
    *,
    pocket_bounds: torch.Tensor | None,
    placement: RandomPlacement,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample random position, uniform SO(3) orientation, and full torsions.

    Position means the ligand heavy-atom centroid.  It is uniform along each
    axis of the pocket box.  All logical torsions are uniform on ``(-pi, pi]``.
    """

    genes = layout.genes_from_pose_stack(layout.template)
    npool = genes.shape[0]
    bounds = _normalize_pocket_bounds(
        layout,
        pocket_bounds=pocket_bounds,
        placement=placement,
    )
    dtype, device = genes.dtype, genes.device
    unit_position = torch.rand(
        (npool, 3), dtype=dtype, device=device, generator=generator
    )
    target_centroid = bounds[0] + unit_position * (bounds[1] - bounds[0])
    result = genes.clone()

    random_values = torch.rand(
        (npool, 3), dtype=dtype, device=device, generator=generator
    )
    u1, u2, u3 = torch.unbind(random_values, dim=1)
    u2 = 2 * math.pi * u2
    u3 = 2 * math.pi * u3
    quaternion = torch.stack(
        (
            torch.sqrt(u1) * torch.cos(u3),
            torch.sqrt(1 - u1) * torch.sin(u2),
            torch.sqrt(1 - u1) * torch.cos(u2),
            torch.sqrt(u1) * torch.sin(u3),
        ),
        dim=1,
    )
    result[:, 3:6] = _rotation_matrix_to_zyx(
        _quaternion_rotation_matrix(quaternion).transpose(-1, -2)
    )
    if result.shape[1] > 6:
        result[:, 6:] = (
            2
            * math.pi
            * torch.rand(
                (npool, result.shape[1] - 6),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            - math.pi
        )
    result = _canonicalize_genes(result)
    oriented = layout.pose_stack_from_genes(result)
    current_centroid = layout.ligand_heavy_coords(oriented).mean(dim=1)
    result[:, :3] += target_centroid - current_centroid
    return result


def _conformer_initial_genes(
    layout: LigandGeneLayout,
    conformer_coords: numpy.ndarray,
    *,
    seed_fraction: float,
    pocket_bounds: torch.Tensor | None,
    placement: RandomPlacement,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace a fraction of blind ligand torsions with conformer torsions."""

    random_genes = _random_initial_genes(
        layout,
        pocket_bounds=pocket_bounds,
        placement=placement,
        generator=generator,
    )
    npool = random_genes.shape[0]
    n_seeded = min(int(math.floor(seed_fraction * npool)), npool - 1)
    if n_seeded == 0:
        return random_genes

    library = torch.as_tensor(
        conformer_coords,
        dtype=layout.template.coords.dtype,
        device=layout.template.device,
    )

    seeded_rows = torch.arange(1, n_seeded + 1, device=layout.template.device)
    conformer_indices = (seeded_rows - 1) % library.shape[0]
    atom_indices = (
        layout.template.block_coord_offset64[
            seeded_rows, layout.docking_dofs.ligand_block, None
        ]
        + layout.ligand_local_atoms[None, :]
    )
    coords = layout.template.coords.clone()
    coords[seeded_rows[:, None], atom_indices] = library[conformer_indices]
    conformer_pose = attr.evolve(layout.template, coords=coords)
    conformer_genes = layout.genes_from_pose_stack(conformer_pose)

    result = random_genes.clone()
    n_ligand_torsions = len(layout.docking_dofs.torsions)
    result[seeded_rows, 6 : 6 + n_ligand_torsions] = conformer_genes[
        seeded_rows, 6 : 6 + n_ligand_torsions
    ]

    random_pose = layout.pose_stack_from_genes(random_genes)
    target_centroids = layout.ligand_heavy_coords(random_pose).mean(dim=1)
    seeded_pose = layout.pose_stack_from_genes(result)
    current_centroids = layout.ligand_heavy_coords(seeded_pose).mean(dim=1)
    result[seeded_rows, :3] += (
        target_centroids[seeded_rows] - current_centroids[seeded_rows]
    )
    return result


def _ranked_result(
    reference_pose_stack: PoseStack,
    pool: GAPool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PoseStack]:
    """Rank the candidate population and construct the single best pose.

    Ranked tensors keep a leading singleton dimension, matching the single
    reference pose: energies are ``[1, npool]`` and genes ``[1, npool, ndof]``.
    """

    order = torch.argsort(pool.energies)
    ranked_energies = pool.energies[order].unsqueeze(0)
    ranked_genes = pool.genes[order].unsqueeze(0)
    best_pose_stack = attr.evolve(
        reference_pose_stack,
        coords=pool.pose_stack.coords[order[:1]].detach().clone(),
    )
    return (
        ranked_genes,
        ranked_energies,
        ranked_genes[:, 0],
        ranked_energies[:, 0],
        best_pose_stack,
    )


def _conformer_library_or_none(ligand_mol2, pose_stack, ligand_block, **library_kwargs):
    """Conformer library for the ligand, or ``None`` (with a warning) if absent."""
    if ligand_mol2 is None:
        logger.warning(
            "conformer initialization requested without ligand_mol2; "
            "falling back to blind random initialization"
        )
        return None
    block_type = pose_stack.block_type(0, ligand_block)
    try:
        return generate_conformer_library(
            ligand_mol2, block_type, pose_stack.packed_block_types, **library_kwargs
        )
    except Exception as exc:
        logger.warning(
            "ligand conformer generation failed (%s: %s); falling "
            "back to blind random initialization",
            type(exc).__name__,
            exc,
        )
        return None


def _initial_genes_for(
    initialization: str,
    layout: LigandGeneLayout,
    backend: ScoringBackend,
    generator: torch.Generator,
    *,
    schedule: GASchedule,
    pocket_bounds: torch.Tensor | None,
    random_placement: RandomPlacement,
    conformers: dict,
    ligand_aligner: LigandAligner,
    pose_stack: PoseStack,
) -> torch.Tensor:
    """Starting genes of the population for the chosen ``initialization``."""

    def random_genes():
        return _random_initial_genes(
            layout,
            pocket_bounds=pocket_bounds,
            placement=random_placement,
            generator=generator,
        )

    if initialization == "jitter":
        return _initial_genes(layout, schedule.initial_perturbation, generator)
    if initialization == "random":
        return random_genes()
    if initialization == "conformer":
        library = _conformer_library_or_none(
            conformers["ligand_mol2"],
            pose_stack,
            0,
            n_conformers=conformers["n_conformers"],
            energy_window=conformers["energy_window"],
            cluster_rmsd=conformers["cluster_rmsd"],
        )
        if library is None:
            return random_genes()
        return _conformer_initial_genes(
            layout,
            library.coords,
            seed_fraction=conformers["seed_fraction"],
            pocket_bounds=pocket_bounds,
            placement=random_placement,
            generator=generator,
        )
    return ligand_aligner.seed_genes(
        layout,
        backend,
        random_genes(),
        pocket_bounds=pocket_bounds,
        generator=generator,
    )


def run_docking(
    pose_stack: PoseStack,
    backend: ScoringBackend,
    *,
    schedule: GASchedule = GASchedule(),
    seed: int = 0,
    initialization: Literal["random", "jitter", "conformer", "aligner"] = "random",
    pocket_bounds: torch.Tensor | None = None,
    random_placement: RandomPlacement = RandomPlacement(),
    ligand_mol2: str | Path | None = None,
    n_conformers: int = 200,
    seed_fraction: float = 0.5,
    energy_window: float = 20.0,
    cluster_rmsd: float = 0.5,
    ligand_aligner: LigandAligner = LigandAligner(),
    finalize: bool = False,
    top_k: int = 1,
) -> DockingResult:
    """Dock the backend's ligand into the complex ``pose_stack``.

    ``pose_stack`` is the single-pose complex the backend was built from, with
    the ligand in any placement. The backend chooses the ligand and which
    receptor sidechains are flexible; a genetic algorithm (``schedule``) then
    searches ligand pose, torsions and flexible chis.

    ``initialization`` selects the starting population: ``"random"`` places the
    ligand randomly (inside ``pocket_bounds`` if given), ``"jitter"`` perturbs
    the input pose (refinement), ``"conformer"`` seeds torsions from a
    conformer library of ``ligand_mol2``, ``"aligner"`` uses ``ligand_aligner``.

    Without ``finalize`` the result holds the grid-ranked pool. With
    ``finalize=True`` the ``top_k`` candidates are promoted to full complexes,
    exact-refined, and ranked by binding free energy (``dG``); ``best_pose_stack``
    and the energies then come from that refinement, while ``best_genes`` /
    ``ranked_genes`` stay the grid-GA genes (Cartesian refinement changes bond
    lengths that torsion genes cannot represent). The grid-GA output is always
    available as ``ga_result``.
    """

    if pose_stack.n_poses != 1:
        raise ValueError("pose_stack must contain one reference pose")
    if initialization == "aligner" and not isinstance(backend, GridBackend):
        raise ValueError("aligner initialization requires a GridBackend")
    reference = pose_stack
    search_pose, search_backend, compact_sidechains = backend.search_view(reference)
    population = repeat_pose_stack(search_pose, schedule.npool)
    ligand_block = search_backend.ligand_block
    docking_dofs = build_ligand_docking_dofs(
        population, ligand_block, sidechain_blocks=compact_sidechains
    )
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)

    genes = _initial_genes_for(
        initialization,
        layout,
        search_backend,
        generator,
        schedule=schedule,
        pocket_bounds=pocket_bounds,
        random_placement=random_placement,
        conformers=dict(
            ligand_mol2=ligand_mol2,
            n_conformers=n_conformers,
            seed_fraction=seed_fraction,
            energy_window=energy_window,
            cluster_rmsd=cluster_rmsd,
        ),
        ligand_aligner=ligand_aligner,
        pose_stack=search_pose,
    )
    if initialization in ("random", "conformer") and compact_sidechains:
        # Keep the reference as the first flexible candidate, making the
        # flexible search monotone w.r.t. the supplied ligand/sidechain state.
        genes[0] = layout.genes_from_pose_stack(layout.template)[0]
    pool = optimize_generation(genes, layout, search_backend, maxiter=schedule.maxiter)
    best_history = [torch.min(pool.energies)]
    diverse_history: list[int] = []

    # Rosetta GAOptimizer.cc:104-113.
    for _ in range(schedule.generations):
        offspring = optimize_generation(
            next_generation(
                pool.genes, schedule.npool, schedule.pmut, generator=generator
            ),
            layout,
            search_backend,
            maxiter=schedule.maxiter,
        )
        pool = update_pool(
            pool,
            offspring,
            layout,
            npool=schedule.npool,
            rmsdthreshold=schedule.rmsdthreshold,
        )
        best_history.append(torch.min(pool.energies))
        diverse_history.append(pool.n_diverse)

    ranked_genes, ranked_energies, best_genes, best_energies, best_pose_stack = (
        _ranked_result(search_pose, pool)
    )
    if search_pose is not reference:
        best_pose_stack = _place_ligand_in_reference(
            reference,
            backend.ligand_block,
            best_pose_stack,
            sidechain_blocks=backend.sidechain_blocks,
            compact_sidechain_blocks=compact_sidechains,
            sidechain_atom_indices=(
                layout.sidechain_atom_indices[0] if compact_sidechains else None
            ),
        )
    ga_result = GAResult(
        pool=pool,
        best_energy_history=torch.stack(best_history),
        diverse_count_history=tuple(diverse_history),
        ranked_genes=ranked_genes,
        ranked_energies=ranked_energies,
        best_genes=best_genes,
        best_energies=best_energies,
        best_pose_stack=best_pose_stack,
    )
    if not finalize:
        return DockingResult(
            best_pose_stack=ga_result.best_pose_stack,
            best_energies=ga_result.best_energies,
            best_genes=ga_result.best_genes,
            ranked_energies=ga_result.ranked_energies,
            ranked_genes=ga_result.ranked_genes,
            ga_result=ga_result,
        )

    from tmol.dock.finalization import finalize_ga_docking

    finalization = finalize_ga_docking(reference, backend, ga_result, top_k=top_k)
    return DockingResult(
        best_pose_stack=attr.evolve(
            reference, coords=finalization.pose_stack.coords[:1].detach().clone()
        ),
        best_energies=finalization.dG[:, 0],
        best_genes=finalization.genes[:, 0],
        ranked_energies=finalization.dG,
        ranked_genes=finalization.genes,
        ga_result=ga_result,
        finalization=finalization,
    )


__all__ = [
    "GAPool",
    "GAResult",
    "GASchedule",
    "DockingResult",
    "LigandGeneLayout",
    "RandomPlacement",
    "crossover",
    "ligand_heavy_atom_rms",
    "mutate",
    "next_generation",
    "optimize_generation",
    "run_docking",
    "select_pocket_residues",
    "update_pool",
]
