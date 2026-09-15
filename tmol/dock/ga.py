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
from tmol.dock.dock_loop import DockingPerturbation, _repeat_pose_stack
from tmol.dock.ligand_aligner import LigandAligner
from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    _block_has_nonpolymer_connection,
    _block_kinforest_nodes,
    _inverse_dofs,
    _ligand_kinforest_nodes,
    _sidechain_coordinate_metadata,
    _sidechain_logical_torsions_from_coords,
    _validated_sidechain_blocks,
    _wrap_angle,
    build_ligand_docking_dofs,
)
from tmol.dock.scoring_backend import (
    GridBackend,
    MinimizationResult,
    ScoringBackend,
)
from tmol.kinematics.datatypes import JumpDOFTypes, NodeType
from tmol.kinematics.move_map import MinimizerMap
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.pose.pdb_info import PDBInfo
from tmol.pose.pose_stack import PoseStack

if TYPE_CHECKING:
    from tmol.dock.finalization import DockingFinalization


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GASchedule:
    """Single-stage VSX schedule for the continuous backend."""

    generations: int = 5
    npool: int = 50
    rmsdthreshold: float = 2.0
    pmut: float = 0.2
    maxiter: int = 50
    initial_perturbation: DockingPerturbation = DockingPerturbation()

    def __post_init__(self) -> None:
        for name in ("generations", "npool", "maxiter"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.rmsdthreshold) or self.rmsdthreshold < 0:
            raise ValueError("rmsdthreshold must be finite and nonnegative")
        if not math.isfinite(self.pmut) or not 0.0 <= self.pmut <= 1.0:
            raise ValueError("pmut must be between zero and one")


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
        if (
            not math.isfinite(self.pocket_padding_angstrom)
            or self.pocket_padding_angstrom < 0
        ):
            raise ValueError(
                "pocket_padding_angstrom must be finite and nonnegative"
            )


@dataclass(frozen=True)
class LigandGeneLayout:
    """Map ``[pose, jump | ligand chis | receptor chis]`` to coordinates."""

    template: PoseStack
    docking_dofs: LigandDockingDOFs
    kin_module: PoseStackKinematicsModule
    base_raw_dofs: torch.Tensor
    jump_nodes: torch.Tensor
    torsion_nodes: torch.Tensor
    torsion_dof_types: torch.Tensor
    torsion_signs: torch.Tensor
    torsion_offsets: torch.Tensor
    ligand_nodes: torch.Tensor
    ligand_local_atoms: torch.Tensor
    heavy_local_atoms: torch.Tensor
    sidechain_nodes: torch.Tensor
    sidechain_atom_indices: torch.Tensor
    sidechain_atom_influence: torch.Tensor

    @classmethod
    def from_pose_stack(
        cls, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> "LigandGeneLayout":
        kin_module = PoseStackKinematicsModule(
            pose_stack, docking_dofs.fold_forest
        )
        kmd = kin_module.kmd
        raw_dofs = _inverse_dofs(pose_stack, kin_module)
        minmap = MinimizerMap(pose_stack, kmd, docking_dofs.move_map)

        jump_mask = (
            (kmd.forest.doftype == int(NodeType.jump))
            & torch.all(
                minmap.dof_mask[:, : int(JumpDOFTypes.RBdel_gamma) + 1],
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
        all_torsions = docking_dofs.torsions + docking_dofs.sidechain_torsions
        raw_atoms = torch.tensor(
            [mapping.raw_atom for mapping in docking_dofs.torsions],
            dtype=torch.int64,
            device=pose_stack.device,
        )
        ligand_torsion_nodes = ligand_nodes[:, raw_atoms]
        if docking_dofs.sidechain_torsions:
            sidechain_torsion_nodes = torch.stack(
                [
                    _block_kinforest_nodes(
                        pose_stack,
                        mapping.block,
                        pose_stack.packed_block_types.active_block_types[
                            int(pose_stack.block_type_ind64[0, mapping.block])
                        ].n_atoms,
                        kmd.forest.id,
                    )[:, mapping.raw_atom]
                    for mapping in docking_dofs.sidechain_torsions
                ],
                dim=1,
            )
            torsion_nodes = torch.cat(
                (ligand_torsion_nodes, sidechain_torsion_nodes), dim=1
            )
        else:
            torsion_nodes = ligand_torsion_nodes
        torsion_dof_types = torch.tensor(
            [int(mapping.dof_type) for mapping in all_torsions],
            dtype=torch.int64,
            device=pose_stack.device,
        )
        torsion_signs = torch.tensor(
            [mapping.sign for mapping in all_torsions],
            dtype=raw_dofs.dtype,
            device=pose_stack.device,
        )
        if all_torsions:
            torsion_offsets = torch.stack(
                [mapping.offset for mapping in all_torsions], dim=1
            ).to(dtype=raw_dofs.dtype)
        else:
            torsion_offsets = torch.empty(
                (pose_stack.n_poses, 0),
                dtype=raw_dofs.dtype,
                device=pose_stack.device,
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
            jump_nodes=jump_nodes,
            torsion_nodes=torsion_nodes,
            torsion_dof_types=torsion_dof_types,
            torsion_signs=torsion_signs,
            torsion_offsets=torsion_offsets,
            ligand_nodes=ligand_nodes,
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

    @property
    def n_dof(self) -> int:
        return 6 + self.torsion_nodes.shape[1]

    def genes_from_raw_dofs(self, raw_dofs: torch.Tensor) -> torch.Tensor:
        """Gather the fixed-order gene from a raw kinematic DOF tensor."""

        if raw_dofs.shape != self.base_raw_dofs.shape:
            raise ValueError(
                "raw DOF shape does not match the gene layout: "
                f"{tuple(raw_dofs.shape)} != {tuple(self.base_raw_dofs.shape)}"
            )
        jump_dofs = raw_dofs[
            self.jump_nodes[:, None],
            torch.arange(6, dtype=torch.int64, device=raw_dofs.device)[None, :],
        ]
        raw_torsions = raw_dofs[
            self.torsion_nodes, self.torsion_dof_types[None, :]
        ]
        logical_torsions = _wrap_angle(
            self.torsion_signs[None, :] * raw_torsions
            + self.torsion_offsets
        )
        return torch.cat((jump_dofs, logical_torsions), dim=1)

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
        if genes.device != self.template.device:
            raise ValueError("genes and gene layout must be on the same device")
        if genes.dtype != self.base_raw_dofs.dtype:
            raise ValueError("genes and kinematic DOFs must have the same dtype")

        raw_dofs = self.base_raw_dofs.clone()
        raw_dofs[
            self.jump_nodes[:, None],
            torch.arange(6, dtype=torch.int64, device=genes.device)[None, :],
        ] = genes[:, :6]
        raw_torsions = _wrap_angle(
            self.torsion_signs[None, :]
            * (genes[:, 6:] - self.torsion_offsets)
        )
        raw_dofs[
            self.torsion_nodes, self.torsion_dof_types[None, :]
        ] = raw_torsions
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
            self.template.block_coord_offset64[
                :, self.docking_dofs.ligand_block, None
            ]
            + self.ligand_local_atoms[None, :]
        )
        coords[pose_indices[:, None], atom_indices] = kin_coords[self.ligand_nodes].to(
            coords.dtype
        )
        if self.sidechain_nodes.shape[1]:
            rebuilt_sidechains = kin_coords[self.sidechain_nodes].to(coords.dtype)
            coords[
                pose_indices[:, None], self.sidechain_atom_indices
            ] = rebuilt_sidechains
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
        pose_stack.block_coord_offset64[:, block, None]
        + heavy_local_atoms[None, :]
    )
    return pose_stack.coords[pose_indices[:, None], atom_indices]


def select_pocket_residue_blocks(
    pose_stack: PoseStack,
    ligand_block: int,
    cutoff: float = 6.0,
) -> torch.Tensor:
    """Return sorted polymer blocks contacting ligand heavy atoms.

    A block is selected when any of its heavy atoms is within ``cutoff`` of a
    ligand heavy atom in any pose. Non-polymer blocks, including cofactors and
    solvent, are excluded. Polymer residues carrying a non-polymer covalent
    connection (e.g. a disulfide-bonded cysteine) are also excluded, because
    flexible-sidechain kinematics cannot preserve that bond; passing such a
    residue explicitly is rejected downstream by ``_validated_sidechain_blocks``.
    Chi-less residues (e.g. ALA, GLY) are excluded as well: they have nothing to
    move, so the returned set is directly usable as flexible ``sidechain_blocks``.
    """

    if not isinstance(ligand_block, int) or isinstance(ligand_block, bool):
        raise TypeError("ligand_block must be an integer")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError("ligand_block is out of range")
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool):
        raise TypeError("cutoff must be a real number")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative")

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
        if _block_has_nonpolymer_connection(
            pose_stack, block, int(block_type_indices[0])
        ):
            continue
        if int(block_type.n_sc_torsions) == 0:
            continue
        receptor_heavy = _block_heavy_coords(pose_stack, block)
        if bool(torch.any(torch.cdist(receptor_heavy, ligand_heavy) <= cutoff)):
            selected.append(block)
    return torch.tensor(
        selected, dtype=torch.int64, device=pose_stack.device
    )


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

    @property
    def finalized_pose_stack(self) -> PoseStack | None:
        return None if self.finalization is None else self.finalization.pose_stack

    @property
    def pre_refine_energies(self) -> torch.Tensor | None:
        return (
            None if self.finalization is None else self.finalization.pre_refine_energies
        )

    @property
    def complex_energies(self) -> torch.Tensor | None:
        return None if self.finalization is None else self.finalization.complex_energies

    @property
    def free_ligand_energies(self) -> torch.Tensor | None:
        return (
            None
            if self.finalization is None
            else self.finalization.free_ligand_energies
        )

    @property
    def free_receptor_energies(self) -> torch.Tensor | None:
        return (
            None
            if self.finalization is None
            else self.finalization.free_receptor_energies
        )

    @property
    def nchi(self) -> torch.Tensor | None:
        return None if self.finalization is None else self.finalization.nchi

    @property
    def dH(self) -> torch.Tensor | None:
        return None if self.finalization is None else self.finalization.dH

    @property
    def TdS(self) -> torch.Tensor | None:
        return None if self.finalization is None else self.finalization.TdS

    @property
    def dG(self) -> torch.Tensor | None:
        return None if self.finalization is None else self.finalization.dG


def _ligand_only_pose_stack(pose_stack: PoseStack, ligand_block: int) -> PoseStack:
    """Select a disconnected ligand block into a single-block PoseStack.

    Coordinates and ligand metadata are copied exactly from each input pose.
    Inter-block connections are empty, the retained diagonal bond-separation
    table is 1x1, and the ligand is assigned to the only chain. Constraints are
    omitted because the grid backend does not score them; the original
    constraint set remains on the full-complex result PoseStack.
    """

    if not isinstance(ligand_block, int) or isinstance(ligand_block, bool):
        raise TypeError("ligand_block must be an integer")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError("ligand_block is out of range")
    if pose_stack.fragmented_ligand_mapping is not None:
        raise ValueError(
            "ligand-only docking does not support fragmented-ligand metadata"
        )

    block_type_indices = pose_stack.block_type_ind64[:, ligand_block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError("ligand_block is absent from at least one pose")
    if not bool(torch.all(block_type_indices == block_type_indices[0])):
        raise ValueError("ligand-only docking requires one homogeneous ligand type")
    block_type_index = int(block_type_indices[0])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    n_atoms = block_type.n_atoms

    source_connections = pose_stack.inter_residue_connections64[
        :, ligand_block : ligand_block + 1
    ]
    if bool(torch.any(source_connections[..., 0] >= 0)):
        raise ValueError(
            "ligand-only grid docking requires a disconnected ligand block"
        )

    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    atom_indices = (
        pose_stack.block_coord_offset64[:, ligand_block, None]
        + torch.arange(n_atoms, dtype=torch.int64, device=pose_stack.device)[None, :]
    )
    coords = pose_stack.coords[pose_indices[:, None], atom_indices].clone()
    block_coord_offset = torch.zeros(
        (pose_stack.n_poses, 1), dtype=torch.int32, device=pose_stack.device
    )
    inter_residue_connections64 = torch.full_like(source_connections, -1)
    inter_block_bondsep64 = pose_stack.inter_block_bondsep64[
        :,
        ligand_block : ligand_block + 1,
        ligand_block : ligand_block + 1,
    ].clone()
    block_type_ind64 = block_type_indices[:, None].clone()
    chain_id64 = torch.zeros_like(block_type_ind64)

    source_atom_offsets = (
        pose_stack.block_coord_offset64[:, ligand_block].detach().cpu().numpy()
    )
    pdb_atom_indices = source_atom_offsets[:, None] + numpy.arange(n_atoms)[None, :]
    pdb_pose_indices = numpy.arange(pose_stack.n_poses)[:, None]
    pdb_info = PDBInfo(
        residue_labels=pose_stack.pdb_info.residue_labels[
            :, ligand_block : ligand_block + 1
        ].copy(),
        residue_insertion_codes=pose_stack.pdb_info.residue_insertion_codes[
            :, ligand_block : ligand_block + 1
        ].copy(),
        chain_labels=pose_stack.pdb_info.chain_labels[
            :, ligand_block : ligand_block + 1
        ].copy(),
        atom_occupancy=pose_stack.pdb_info.atom_occupancy[
            pdb_pose_indices, pdb_atom_indices
        ].copy(),
        atom_b_factor=pose_stack.pdb_info.atom_b_factor[
            pdb_pose_indices, pdb_atom_indices
        ].copy(),
    )

    return PoseStack(
        packed_block_types=pose_stack.packed_block_types,
        coords=coords,
        block_coord_offset=block_coord_offset,
        block_coord_offset64=block_coord_offset.to(torch.int64),
        inter_residue_connections=inter_residue_connections64.to(torch.int32),
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
        inter_block_bondsep64=inter_block_bondsep64,
        block_type_ind=block_type_ind64.to(torch.int32),
        block_type_ind64=block_type_ind64,
        chain_id=chain_id64.to(torch.int32),
        chain_id64=chain_id64,
        pdb_info=pdb_info,
        constraint_set=None,
        device=pose_stack.device,
    )


def _movable_pose_stack(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> PoseStack:
    """Select ligand, movable residues, and peptide context into a compact pose.

    The ligand is block 0, remains disconnected, and receives a compact chain
    id distinct from every receptor chain so it is always an independent
    kinematic root. Selected receptor blocks and their unique immediate peptide
    neighbors follow together in ascending source-block order, preserving the
    contiguous block ranges required by polymer fold-forest edges. Connections
    between retained polymer residues are remapped, while connections to
    omitted residues are cleared. Passing no sidechain blocks delegates to the
    historical ligand-only builder exactly.
    """

    sidechain_blocks = _validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    if not sidechain_blocks:
        return _ligand_only_pose_stack(pose_stack, ligand_block)
    if pose_stack.fragmented_ligand_mapping is not None:
        raise ValueError(
            "movable grid docking does not support fragmented-ligand metadata"
        )
    if not bool(
        torch.all(pose_stack.block_type_ind64 == pose_stack.block_type_ind64[0:1])
    ):
        raise ValueError("movable grid docking requires homogeneous topology")
    if bool(
        torch.any(pose_stack.inter_residue_connections64[:, ligand_block, :, 0] >= 0)
    ):
        raise ValueError("movable grid docking requires a disconnected ligand block")

    source_blocks, _, _ = _movable_block_mapping(
        pose_stack, ligand_block, sidechain_blocks
    )
    keep_blocks = torch.tensor(
        source_blocks,
        dtype=torch.int64,
        device=pose_stack.device,
    )
    old_to_new = torch.full(
        (pose_stack.max_n_blocks,),
        -1,
        dtype=torch.int64,
        device=pose_stack.device,
    )
    old_to_new[keep_blocks] = torch.arange(
        keep_blocks.numel(), dtype=torch.int64, device=pose_stack.device
    )

    connections64 = pose_stack.inter_residue_connections64[:, keep_blocks].clone()
    partner_blocks = connections64[..., 0]
    real_partners = partner_blocks >= 0
    mapped_partners = torch.full_like(partner_blocks, -1)
    mapped_partners[real_partners] = old_to_new[partner_blocks[real_partners]]
    omitted_partners = real_partners & (mapped_partners < 0)
    connections64[omitted_partners] = -1
    retained_partners = mapped_partners >= 0
    partner_blocks[retained_partners] = mapped_partners[retained_partners]

    block_type_ind64 = pose_stack.block_type_ind64[:, keep_blocks].clone()
    pbt = pose_stack.packed_block_types
    atom_counts = pbt.n_atoms[block_type_ind64[0]].to(torch.int64)
    offsets64 = torch.zeros_like(block_type_ind64)
    offsets64[:, 1:] = torch.cumsum(atom_counts[:-1], dim=0)[None, :]

    source_atom_parts = []
    for source_block, n_atoms in zip(keep_blocks, atom_counts):
        start = pose_stack.block_coord_offset64[0, source_block]
        source_atom_parts.append(
            start
            + torch.arange(
                int(n_atoms), dtype=torch.int64, device=pose_stack.device
            )
        )
    source_atoms = torch.cat(source_atom_parts)
    coords = pose_stack.coords[:, source_atoms].clone()

    keep_host = keep_blocks.detach().cpu().numpy()
    source_host = source_atoms.detach().cpu().numpy()
    pdb_info = PDBInfo(
        residue_labels=pose_stack.pdb_info.residue_labels[:, keep_host].copy(),
        residue_insertion_codes=(
            pose_stack.pdb_info.residue_insertion_codes[:, keep_host].copy()
        ),
        chain_labels=pose_stack.pdb_info.chain_labels[:, keep_host].copy(),
        atom_occupancy=pose_stack.pdb_info.atom_occupancy[:, source_host].copy(),
        atom_b_factor=pose_stack.pdb_info.atom_b_factor[:, source_host].copy(),
    )
    inter_block_bondsep64 = pose_stack.inter_block_bondsep64[
        :, keep_blocks[:, None], keep_blocks[None, :]
    ].clone()
    chain_id64 = pose_stack.chain_id64[:, keep_blocks].clone()
    # Source chain ids are compact indices. One past the retained receptor
    # maximum preserves all receptor relationships and cannot collide with any.
    chain_id64[:, 0] = torch.amax(chain_id64[:, 1:], dim=1) + 1

    return PoseStack(
        packed_block_types=pbt,
        coords=coords,
        block_coord_offset=offsets64.to(torch.int32),
        block_coord_offset64=offsets64,
        inter_residue_connections=connections64.to(torch.int32),
        inter_residue_connections64=connections64,
        inter_block_bondsep=inter_block_bondsep64.to(torch.int32),
        inter_block_bondsep64=inter_block_bondsep64,
        block_type_ind=block_type_ind64.to(torch.int32),
        block_type_ind64=block_type_ind64,
        chain_id=chain_id64.to(torch.int32),
        chain_id64=chain_id64,
        pdb_info=pdb_info,
        constraint_set=None,
        device=pose_stack.device,
    )


def _sidechain_context_blocks(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> tuple[int, ...]:
    """Return unique immediate polymer neighbors of selected sidechains.

    Only the residue type's declared lower/upper polymer connections are
    followed; non-polymeric connections such as disulfides are not peptide
    context. Selected sidechains and the ligand are never returned.
    """

    sidechain_blocks = _validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    if not sidechain_blocks:
        return ()

    pbt = pose_stack.packed_block_types
    excluded = {ligand_block, *sidechain_blocks}
    context: set[int] = set()
    for block in sidechain_blocks:
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        connection_indices = (
            int(pbt.down_conn_inds[block_type_index]),
            int(pbt.up_conn_inds[block_type_index]),
        )
        for connection in connection_indices:
            if connection < 0:
                continue
            partners = pose_stack.inter_residue_connections64[
                :, block, connection, 0
            ]
            partner_connections = pose_stack.inter_residue_connections64[
                :, block, connection, 1
            ]
            if bool(torch.any(partners != partners[0])):
                raise ValueError(
                    "movable grid docking requires identical peptide context "
                    "across poses"
                )
            partner = int(partners[0])
            if partner < 0 or partner in excluded:
                continue
            partner_type_indices = pose_stack.block_type_ind64[:, partner]
            if bool(torch.any(partner_type_indices < 0)):
                raise ValueError(
                    "a sidechain peptide neighbor is absent from at least one pose"
                )
            if bool(torch.any(partner_type_indices != partner_type_indices[0])):
                raise ValueError(
                    "all poses must use the same peptide-neighbor block type"
                )
            partner_type = pbt.active_block_types[int(partner_type_indices[0])]
            if not partner_type.properties.polymer.is_polymer:
                raise ValueError(
                    "a declared polymer connection points to a non-polymer block"
                )
            expected_partner_connection = (
                partner_type.up_connection_ind
                if connection == int(pbt.down_conn_inds[block_type_index])
                else partner_type.down_connection_ind
            )
            if bool(
                torch.any(partner_connections != expected_partner_connection)
            ):
                raise ValueError(
                    "a declared polymer connection is not reciprocally peptide-linked"
                )
            context.add(partner)
    return tuple(sorted(context))


def _movable_block_mapping(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return compact source order and flexible/context compact indices."""

    sidechain_blocks = _validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    context_blocks = _sidechain_context_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    receptor_blocks = tuple(sorted(sidechain_blocks + context_blocks))
    source_blocks = (ligand_block,) + receptor_blocks
    source_to_compact = {
        source_block: compact_block
        for compact_block, source_block in enumerate(source_blocks)
    }
    compact_sidechains = tuple(
        source_to_compact[source_block] for source_block in sidechain_blocks
    )
    compact_context = tuple(
        source_to_compact[source_block] for source_block in context_blocks
    )
    return source_blocks, compact_sidechains, compact_context


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
    coords[pose_indices[:, None], atom_indices] = ligand_pose_stack.coords[
        :, :n_atoms
    ]
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
            source_type_indices = reference_pose_stack.block_type_ind64[
                :, source_block
            ]
            if not bool(
                torch.all(
                    source_type_indices
                    == ligand_pose_stack.block_type_ind64[:, compact_block]
                )
            ):
                raise ValueError("sidechain and reference block types must match")
            target_atoms = (
                reference_pose_stack.block_coord_offset64[
                    :, source_block, None
                ]
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
    vectors = torch.randn(
        (count, 3), dtype=dtype, device=device, generator=generator
    )
    return vectors / torch.linalg.vector_norm(
        vectors, dim=1, keepdim=True
    ).clamp_min(torch.finfo(dtype).tiny)


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

    if genes.ndim != 2 or genes.shape[1] < 6:
        raise ValueError("genes must have shape [population, 6+n_torsions]")
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
    perturbation_rotation = _quaternion_rotation_matrix(
        perturbation_quaternion
    )
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

    if genes.ndim != 2 or genes.shape[1] < 6:
        raise ValueError("genes must have shape [population, 6+n_torsions]")
    n_genes, n_dof = genes.shape
    device, dtype = genes.device, genes.dtype
    randomize_rt = (
        torch.rand((n_genes,), device=device, generator=generator)
        < rtmutation_rate
    )
    result = genes.clone()

    translation_axis = _unit_vectors(
        n_genes, dtype=dtype, device=device, generator=generator
    )
    translation_length = (
        torch.rand(
            (n_genes, 1), dtype=dtype, device=device, generator=generator
        )
        * translation_width
    )
    result[:, :3] += (
        randomize_rt[:, None] * translation_axis * translation_length
    )

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
        mutated_rotation = _rotation_matrix_to_zyx(
            random_rotation.transpose(-1, -2)
        )
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
    result[:, 3:6] = torch.where(
        randomize_rt[:, None], mutated_rotation, genes[:, 3:6]
    )

    n_torsions = n_dof - 6
    if n_torsions:
        mutate_torsion = (
            torch.rand(
                (n_genes, n_torsions), device=device, generator=generator
            )
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
            mutated_torsions = 2 * math.pi * torch.rand(
                (n_genes, n_torsions),
                dtype=dtype,
                device=device,
                generator=generator,
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
        result[:, 6:] = torch.where(
            mutate_torsion, mutated_torsions, genes[:, 6:]
        )
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

    if parent1.shape != parent2.shape or parent1.ndim != 2:
        raise ValueError("crossover parents must have matching 2D shapes")
    if parent1.shape[1] < 6:
        raise ValueError("crossover genes need a six-value rigid body")
    n_genes, n_dof = parent1.shape
    take_rb_1 = torch.rand(
        (n_genes, 1), device=parent1.device, generator=generator
    ) < 0.5
    result = torch.empty_like(parent1)
    result[:, :6] = torch.where(take_rb_1, parent1[:, :6], parent2[:, :6])

    n_torsions = n_dof - 6
    if n_torsions:
        take_chi_1 = torch.rand(
            (n_genes, n_torsions),
            device=parent1.device,
            generator=generator,
        ) < 0.5
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
        result[:, 6:] = torch.where(
            take_chi_1, parent1[:, 6:], parent2[:, 6:]
        )
    return _canonicalize_genes(result)


def ligand_heavy_atom_rms(
    coords1: torch.Tensor, coords2: torch.Tensor
) -> torch.Tensor:
    """Unsymmetrized heavy-atom coordinate RMS with broadcasting."""

    if coords1.shape[-2:] != coords2.shape[-2:]:
        raise ValueError("ligand coordinate tensors must have matching atom shapes")
    delta = coords1 - coords2
    return torch.sqrt(torch.mean(torch.sum(delta * delta, dim=-1), dim=-1))


def _diverse_low_energy_indices(
    energies: torch.Tensor,
    coords: torch.Tensor,
    *,
    npool: int,
    rmsdthreshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select energy-ranked RMS niches, then fill from rejected candidates.

    The bounded host loop walks candidate energy ranks; RMS storage is limited
    to the selected pool instead of an atom-expanded all-pairs candidate tensor.
    """

    if energies.ndim != 1:
        raise ValueError("energies must have shape [candidate]")
    if coords.ndim != 3 or coords.shape[0] != energies.shape[0]:
        raise ValueError("coords must have shape [candidate, heavy_atom, 3]")
    if coords.shape[-1] != 3 or coords.shape[1] < 1:
        raise ValueError("coords must contain at least one 3D heavy atom")
    if not isinstance(npool, int) or isinstance(npool, bool) or npool < 1:
        raise ValueError("npool must be a positive integer")
    n_candidates = energies.shape[0]
    if npool > n_candidates:
        raise ValueError("npool cannot exceed the number of candidates")
    if not math.isfinite(rmsdthreshold) or rmsdthreshold < 0:
        raise ValueError("rmsdthreshold must be finite and nonnegative")

    order = torch.argsort(energies)
    selected_order = torch.full((npool,), -1, dtype=torch.int64, device=energies.device)
    selected_mask = torch.zeros(n_candidates, dtype=torch.bool, device=energies.device)
    selected_count = torch.zeros((), dtype=torch.int64, device=energies.device)
    selected_coords = coords[:1].expand(npool, -1, -1).clone()
    pool_columns = torch.arange(npool, device=energies.device)

    for rank in range(n_candidates):
        candidate = order[rank : rank + 1]
        candidate_coords = coords.index_select(0, candidate)
        candidate_distances = ligand_heavy_atom_rms(candidate_coords, selected_coords)
        active_selected = pool_columns < selected_count
        sufficiently_distant = torch.all(
            ~active_selected | (candidate_distances >= rmsdthreshold)
        )
        accept = sufficiently_distant & (selected_count < npool)
        write_column = selected_count.clamp_max(npool - 1).reshape(1)
        previous = selected_order.gather(0, write_column)
        selected_order.scatter_(
            0, write_column, torch.where(accept, candidate, previous)
        )
        selected_mask.scatter_(0, candidate, accept.reshape(1))
        write_coords = write_column[:, None, None].expand(-1, coords.shape[1], 3)
        previous_coords = selected_coords.gather(0, write_coords)
        selected_coords.scatter_(
            0, write_coords, torch.where(accept, candidate_coords, previous_coords)
        )
        selected_count += accept.to(torch.int64)

    selected_in_rank_order = selected_mask.gather(0, order)
    rejected_rank = torch.where(
        ~selected_in_rank_order,
        torch.arange(n_candidates, device=energies.device),
        n_candidates,
    )
    rejected_order = order.gather(0, torch.argsort(rejected_rank))
    fill_columns = (pool_columns - selected_count).clamp_min(0)
    filled = rejected_order.gather(0, fill_columns)
    selected = torch.where(pool_columns < selected_count, selected_order, filled)
    return selected, selected_count


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

    if genes.ndim != 2 or genes.shape[0] < 1:
        raise ValueError("genes must be a nonempty 2D population")
    if not isinstance(npoolout, int) or isinstance(npoolout, bool) or npoolout < 1:
        raise ValueError("npoolout must be a positive integer")
    if not math.isfinite(pmut) or not 0.0 <= pmut <= 1.0:
        raise ValueError("pmut must be between zero and one")
    npoolin = genes.shape[0]
    parent_indices = torch.arange(
        npoolout, dtype=torch.int64, device=genes.device
    ) % npoolin
    parents = genes[parent_indices]
    mutate_choice = torch.rand(
        (npoolout,), device=genes.device, generator=generator
    ) <= pmut
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


def _refine_swarm_pool(
    swarm: GAPool,
    layout: LigandGeneLayout,
    backend: ScoringBackend,
    *,
    maxiter: int,
    rounds: int,
    chi_step: float,
    generator: torch.Generator,
) -> GAPool:
    """Basin-hop ligand torsions while retaining each candidate's best energy."""

    swarm_size = swarm.genes.shape[0]
    n_ligand_torsions = len(layout.docking_dofs.torsions)
    best_genes = swarm.genes
    best_energies = swarm.energies
    for _ in range(rounds):
        unit_kick = torch.rand(
            (swarm_size, n_ligand_torsions),
            dtype=best_genes.dtype,
            device=best_genes.device,
            generator=generator,
        )
        kick = torch.zeros_like(best_genes)
        kick[:, 6 : 6 + n_ligand_torsions] = (2 * unit_kick - 1) * chi_step
        trial_genes = _canonicalize_genes(best_genes + kick)
        trial_pool = optimize_generation(
            trial_genes,
            layout,
            backend,
            maxiter=maxiter,
        )
        improve = trial_pool.energies < best_energies
        best_genes = torch.where(improve[:, None], trial_pool.genes, best_genes)
        best_energies = torch.where(improve, trial_pool.energies, best_energies)

    return GAPool(
        genes=best_genes,
        pose_stack=layout.pose_stack_from_genes(best_genes),
        energies=best_energies,
        n_diverse=swarm.n_diverse,
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
    order = torch.argsort(energies)
    distances = ligand_heavy_atom_rms(coords[:, None], coords[None, :])

    selected: list[int] = []
    selected_mask = torch.zeros(
        genes.shape[0], dtype=torch.bool, device=genes.device
    )
    for candidate in order.detach().cpu().tolist():
        if not selected:
            selected.append(candidate)
            selected_mask[candidate] = True
        else:
            sufficiently_distant = (
                ~selected_mask
                | (distances[candidate] >= rmsdthreshold)
            )
            if bool(torch.all(sufficiently_distant)):
                selected.append(candidate)
                selected_mask[candidate] = True
        if len(selected) == npool:
            break
    n_diverse = len(selected)

    selected_tensor = torch.tensor(
        selected, dtype=torch.int64, device=genes.device
    )
    if n_diverse < npool:
        rejected_in_energy_order = order[~selected_mask[order]]
        selected_tensor = torch.cat(
            (selected_tensor, rejected_in_energy_order[: npool - n_diverse])
        )

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


def _subsample_swarm_pool(
    swarm: GAPool,
    swarm_layout: LigandGeneLayout,
    pool_layout: LigandGeneLayout,
    *,
    swarm_size: int,
    npool: int,
    rmsdthreshold: float,
) -> GAPool:
    """Reduce the wide minimized swarm to a small diverse GA pool."""

    if swarm.genes.shape[0] != swarm_size:
        raise ValueError("swarm population does not match swarm_size")
    if pool_layout.template.n_poses != npool:
        raise ValueError("pool layout does not match npool")

    # Match update_pool: niching coordinates are rebuilt from minimized genes,
    # while retained coordinates are the backend's minimized pose coordinates.
    swarm_from_genes = swarm_layout.pose_stack_from_genes(swarm.genes)
    heavy_coords = swarm_layout.ligand_heavy_coords(swarm_from_genes)
    selected, n_diverse = _diverse_low_energy_indices(
        swarm.energies,
        heavy_coords,
        npool=npool,
        rmsdthreshold=rmsdthreshold,
    )
    return GAPool(
        genes=swarm.genes[selected],
        pose_stack=attr.evolve(
            pool_layout.template, coords=swarm.pose_stack.coords[selected]
        ),
        energies=swarm.energies[selected],
        n_diverse=int(n_diverse),
        selected_indices=selected,
    )


def _initial_genes(
    layout: LigandGeneLayout,
    perturbation: DockingPerturbation,
    generator: torch.Generator,
) -> torch.Tensor:
    genes = layout.genes_from_pose_stack(layout.template)
    unit_jitter = 2 * torch.rand(
        genes.shape,
        dtype=genes.dtype,
        device=genes.device,
        generator=generator,
    ) - 1
    jitter_rows = torch.arange(genes.shape[0], device=genes.device) > 0
    result = genes.clone()
    result[:, :3] += (
        jitter_rows[:, None]
        * unit_jitter[:, :3]
        * perturbation.translation_angstrom
    )
    result[:, 3:6] += (
        jitter_rows[:, None]
        * unit_jitter[:, 3:6]
        * perturbation.rotation_radians
    )
    result[:, 6:] += (
        jitter_rows[:, None]
        * unit_jitter[:, 6:]
        * perturbation.torsion_radians
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
    if bounds.shape != (2, 3):
        raise ValueError("pocket_bounds must have shape [2, 3]")
    if not torch.all(torch.isfinite(bounds)) or not torch.all(
        bounds[1] >= bounds[0]
    ):
        raise ValueError("pocket bounds must be finite with max >= min")
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
        result[:, 6:] = 2 * math.pi * torch.rand(
            (npool, result.shape[1] - 6),
            dtype=dtype,
            device=device,
            generator=generator,
        ) - math.pi
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
    if not math.isfinite(seed_fraction) or not 0.0 <= seed_fraction <= 1.0:
        raise ValueError("seed_fraction must be between zero and one")
    npool = random_genes.shape[0]
    n_seeded = min(int(math.floor(seed_fraction * npool)), npool - 1)
    if n_seeded == 0:
        return random_genes

    library = torch.as_tensor(
        conformer_coords,
        dtype=layout.template.coords.dtype,
        device=layout.template.device,
    )
    n_atoms = layout.ligand_local_atoms.shape[0]
    if library.ndim != 3 or library.shape[1:] != (n_atoms, 3):
        raise ValueError(
            "conformer coordinates must have shape [conformer, ligand_atom, 3]"
        )
    if library.shape[0] < 1 or not torch.all(torch.isfinite(library)):
        raise ValueError("conformer coordinates must be nonempty and finite")

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
    result[
        seeded_rows, 6 : 6 + n_ligand_torsions
    ] = conformer_genes[seeded_rows, 6 : 6 + n_ligand_torsions]

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


def run_ga_docking(
    pose_stack: PoseStack,
    ligand_block: int,
    backend: ScoringBackend,
    *,
    schedule: GASchedule = GASchedule(),
    seed: int = 0,
    initialization: Literal[
        "jitter", "random", "conformer", "aligner", "swarm"
    ] = "jitter",
    swarm_size: int = 4000,
    swarm_mc_rounds: int = 0,
    swarm_chi_step: float = 0.52359878,
    pocket_bounds: torch.Tensor | None = None,
    random_placement: RandomPlacement = RandomPlacement(),
    sidechain_blocks: tuple[int, ...] | None = None,
    ligand_mol2: str | Path | None = None,
    n_conformers: int = 200,
    seed_fraction: float = 0.5,
    energy_window: float = 20.0,
    cluster_rmsd: float = 0.5,
    ligand_aligner: LigandAligner = LigandAligner(),
) -> GAResult:
    """Dock one reference ligand using a population of candidate poses.

    Rigid grid docking builds its population from a ligand-only single-block
    view. Flexible grid docking uses a compact ligand-plus-sidechains view;
    both restore the winning movable coordinates to the single input complex.
    Swarm initialization minimizes a wide random population before selecting
    the GA pool. Optional MC rounds kick ligand torsions uniformly within
    ``[-swarm_chi_step, swarm_chi_step)`` radians, minimize, and retain each
    candidate's best energy before selection.
    Conformer initialization seeds ligand torsions from a library.
    Aligner sampling and seed-fraction controls are carried by
    ``ligand_aligner``; the scalar ``seed_fraction`` applies only to conformer
    initialization.
    """

    if pose_stack.n_poses != 1:
        raise ValueError("pose_stack must contain one reference pose")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if initialization not in (
        "jitter",
        "random",
        "conformer",
        "aligner",
        "swarm",
    ):
        raise ValueError(
            "initialization must be 'jitter', 'random', 'conformer', 'aligner', "
            "or 'swarm'"
        )
    if not isinstance(random_placement, RandomPlacement):
        raise TypeError("random_placement must be a RandomPlacement")
    if not isinstance(ligand_aligner, LigandAligner):
        raise TypeError("ligand_aligner must be a LigandAligner")
    if initialization == "aligner" and not isinstance(backend, GridBackend):
        raise ValueError("aligner initialization requires a GridBackend")
    if initialization == "swarm" and (
        not isinstance(swarm_size, int)
        or isinstance(swarm_size, bool)
        or swarm_size < schedule.npool
    ):
        raise ValueError("swarm_size must be an integer at least schedule.npool")
    if initialization == "swarm" and (
        not isinstance(swarm_mc_rounds, int)
        or isinstance(swarm_mc_rounds, bool)
        or swarm_mc_rounds < 0
    ):
        raise ValueError("swarm_mc_rounds must be a nonnegative integer")
    if initialization == "swarm" and (
        not math.isfinite(swarm_chi_step) or swarm_chi_step < 0
    ):
        raise ValueError("swarm_chi_step must be finite and nonnegative")
    complex_reference = None
    complex_ligand_block = ligand_block
    requested_sidechain_blocks = (
        ()
        if sidechain_blocks is None
        else _validated_sidechain_blocks(
            pose_stack, ligand_block, sidechain_blocks
        )
    )
    compact_sidechain_blocks: tuple[int, ...] = ()
    if initialization == "swarm" and requested_sidechain_blocks:
        raise ValueError(
            "swarm initialization is not yet supported with sidechain_blocks"
        )
    if isinstance(backend, GridBackend):
        if backend.ligand_block != ligand_block:
            raise ValueError(
                f"backend was built for ligand block {backend.ligand_block}, "
                f"not {ligand_block}"
            )
        if backend.sidechain_blocks != requested_sidechain_blocks:
            raise ValueError(
                "run_ga_docking sidechain_blocks must match the GridBackend "
                "construction"
            )
        complex_reference = pose_stack
        if requested_sidechain_blocks:
            actual_context = _sidechain_context_blocks(
                pose_stack, ligand_block, requested_sidechain_blocks
            )
            if actual_context != backend.context_blocks:
                raise ValueError(
                    "run_ga_docking peptide context must match the GridBackend "
                    "construction"
                )
            _, compact_sidechain_blocks, _ = _movable_block_mapping(
                pose_stack, ligand_block, requested_sidechain_blocks
            )
            pose_stack = _movable_pose_stack(
                pose_stack, ligand_block, requested_sidechain_blocks
            )
        else:
            pose_stack = _ligand_only_pose_stack(pose_stack, ligand_block)
        ligand_block = 0
        backend = backend.for_ligand_block(ligand_block)
    elif requested_sidechain_blocks:
        raise ValueError(
            "flexible-sidechain docking requires a GridBackend built with "
            "the receptor-minus-flexible-and-neighbor partition"
        )
    population = _repeat_pose_stack(pose_stack, schedule.npool)
    if compact_sidechain_blocks:
        docking_dofs = build_ligand_docking_dofs(
            population,
            ligand_block,
            sidechain_blocks=compact_sidechain_blocks,
        )
    else:
        docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    generator = torch.Generator(device=pose_stack.device)
    generator.manual_seed(seed)
    pool = None

    if initialization == "swarm":
        swarm_population = _repeat_pose_stack(pose_stack, swarm_size)
        swarm_dofs = build_ligand_docking_dofs(swarm_population, ligand_block)
        swarm_layout = LigandGeneLayout.from_pose_stack(
            swarm_population, swarm_dofs
        )
        swarm_genes = _random_initial_genes(
            swarm_layout,
            pocket_bounds=pocket_bounds,
            placement=random_placement,
            generator=generator,
        )
        swarm_pool = optimize_generation(
            swarm_genes,
            swarm_layout,
            backend,
            maxiter=schedule.maxiter,
        )
        if swarm_mc_rounds > 0:
            swarm_pool = _refine_swarm_pool(
                swarm_pool,
                swarm_layout,
                backend,
                maxiter=schedule.maxiter,
                rounds=swarm_mc_rounds,
                chi_step=swarm_chi_step,
                generator=generator,
            )
        pool = _subsample_swarm_pool(
            swarm_pool,
            swarm_layout,
            layout,
            swarm_size=swarm_size,
            npool=schedule.npool,
            rmsdthreshold=schedule.rmsdthreshold,
        )
        del swarm_genes
        del swarm_pool
        del swarm_layout
        del swarm_dofs
        del swarm_population
    elif initialization == "jitter":
        genes = _initial_genes(
            layout, schedule.initial_perturbation, generator
        )
    elif initialization == "random":
        genes = _random_initial_genes(
            layout,
            pocket_bounds=pocket_bounds,
            placement=random_placement,
            generator=generator,
        )
    elif initialization == "conformer":
        if not math.isfinite(seed_fraction) or not 0.0 <= seed_fraction <= 1.0:
            raise ValueError("seed_fraction must be between zero and one")
        if (
            not isinstance(n_conformers, int)
            or isinstance(n_conformers, bool)
            or n_conformers < 1
        ):
            raise ValueError("n_conformers must be a positive integer")
        if not math.isfinite(energy_window) or energy_window < 0:
            raise ValueError("energy_window must be finite and nonnegative")
        if not math.isfinite(cluster_rmsd) or cluster_rmsd < 0:
            raise ValueError("cluster_rmsd must be finite and nonnegative")
        conformer_library = None
        if ligand_mol2 is None:
            logger.warning(
                "conformer initialization requested without ligand_mol2; "
                "falling back to blind random initialization"
            )
        else:
            block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
            block_type = pose_stack.packed_block_types.active_block_types[
                block_type_index
            ]
            try:
                conformer_library = generate_conformer_library(
                    ligand_mol2,
                    block_type,
                    pose_stack.packed_block_types,
                    n_conformers=n_conformers,
                    energy_window=energy_window,
                    cluster_rmsd=cluster_rmsd,
                )
            except Exception as exc:
                logger.warning(
                    "ligand conformer generation failed (%s: %s); falling "
                    "back to blind random initialization",
                    type(exc).__name__,
                    exc,
                )
        if conformer_library is None:
            genes = _random_initial_genes(
                layout,
                pocket_bounds=pocket_bounds,
                placement=random_placement,
                generator=generator,
            )
        else:
            genes = _conformer_initial_genes(
                layout,
                conformer_library.coords,
                seed_fraction=seed_fraction,
                pocket_bounds=pocket_bounds,
                placement=random_placement,
                generator=generator,
            )
    else:
        random_genes = _random_initial_genes(
            layout,
            pocket_bounds=pocket_bounds,
            placement=random_placement,
            generator=generator,
        )
        genes = ligand_aligner.seed_genes(
            layout,
            backend,
            random_genes,
            pocket_bounds=pocket_bounds,
            generator=generator,
        )
    if initialization in ("random", "conformer") and compact_sidechain_blocks:
        # Retain the reference as the first flexible candidate, making
        # the flexible search monotone with respect to the supplied
        # ligand/sidechain state while the remaining members retain the
        # historical random-placement distribution.
        reference_genes = layout.genes_from_pose_stack(layout.template)
        genes[0] = reference_genes[0]
    if pool is None:
        pool = optimize_generation(
            genes,
            layout,
            backend,
            maxiter=schedule.maxiter,
        )
    best_history = [torch.min(pool.energies)]
    diverse_history: list[int] = []

    # Rosetta GAOptimizer.cc:104-113.
    for _ in range(schedule.generations):
        offspring_genes = next_generation(
            pool.genes,
            schedule.npool,
            schedule.pmut,
            generator=generator,
        )
        offspring = optimize_generation(
            offspring_genes, layout, backend, maxiter=schedule.maxiter
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

    (
        ranked_genes,
        ranked_energies,
        best_genes,
        best_energies,
        best_pose_stack,
    ) = _ranked_result(pose_stack, pool)
    if complex_reference is not None:
        best_pose_stack = _place_ligand_in_reference(
            complex_reference,
            complex_ligand_block,
            best_pose_stack,
            sidechain_blocks=requested_sidechain_blocks,
            compact_sidechain_blocks=compact_sidechain_blocks,
            sidechain_atom_indices=(
                layout.sidechain_atom_indices[0]
                if requested_sidechain_blocks
                else None
            ),
        )
    return GAResult(
        pool=pool,
        best_energy_history=torch.stack(best_history),
        diverse_count_history=tuple(diverse_history),
        ranked_genes=ranked_genes,
        ranked_energies=ranked_energies,
        best_genes=best_genes,
        best_energies=best_energies,
        best_pose_stack=best_pose_stack,
    )


def run_docking(
    pose_stack: PoseStack,
    ligand_block: int,
    backend: ScoringBackend,
    *,
    schedule: GASchedule = GASchedule(),
    seed: int = 0,
    pocket_bounds: torch.Tensor | None = None,
    random_placement: RandomPlacement = RandomPlacement(),
    initialization: Literal[
        "jitter", "random", "conformer", "aligner", "swarm"
    ] = "random",
    swarm_size: int = 4000,
    swarm_mc_rounds: int = 0,
    swarm_chi_step: float = 0.52359878,
    finalize: bool = False,
    top_k: int = 1,
    sidechain_blocks: tuple[int, ...] | None = None,
    ligand_mol2: str | Path | None = None,
    n_conformers: int = 200,
    seed_fraction: float = 0.5,
    energy_window: float = 20.0,
    cluster_rmsd: float = 0.5,
    ligand_aligner: LigandAligner = LigandAligner(),
) -> DockingResult:
    """Run one prepared ligand through the selected initialization and GA docking.

    Without finalization, the returned ``best_pose_stack`` contains one pose,
    while the ranked tensors retain all ``schedule.npool`` final candidates with
    a leading singleton dimension. With ``finalize=True``, only the grid-ranked
    ``top_k`` are promoted to full complexes, exact-refined, and returned in
    dG-ranked order; the original grid-GA result remains available as
    ``ga_result``.

    NOTE: with ``finalize=True`` the ``best_pose_stack`` holds the exact
    Cartesian-refined coordinates, but ``best_genes`` / ``ranked_genes`` remain
    the PRE-refinement grid-GA genes. Cartesian refinement changes bond lengths
    and angles that the torsion-space gene cannot represent, so rebuilding
    coordinates from these genes reproduces the pre-refine pose, NOT
    ``best_pose_stack``. Use ``best_pose_stack`` / ``finalization.pose_stack``
    for the refined coordinates and the genes only as grid-GA provenance.
    """

    if not isinstance(finalize, bool):
        raise TypeError("finalize must be bool")
    if finalize and (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
        or top_k > schedule.npool
    ):
        raise ValueError("top_k must be between 1 and schedule.npool")

    ga_result = run_ga_docking(
        pose_stack,
        ligand_block,
        backend,
        schedule=schedule,
        seed=seed,
        initialization=initialization,
        swarm_size=swarm_size,
        swarm_mc_rounds=swarm_mc_rounds,
        swarm_chi_step=swarm_chi_step,
        pocket_bounds=pocket_bounds,
        random_placement=random_placement,
        sidechain_blocks=sidechain_blocks,
        ligand_mol2=ligand_mol2,
        n_conformers=n_conformers,
        seed_fraction=seed_fraction,
        energy_window=energy_window,
        cluster_rmsd=cluster_rmsd,
        ligand_aligner=ligand_aligner,
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

    finalization = finalize_ga_docking(
        pose_stack, ligand_block, backend, ga_result, top_k=top_k
    )
    best_pose_stack = attr.evolve(
        pose_stack,
        coords=finalization.pose_stack.coords[:1].detach().clone(),
    )
    return DockingResult(
        best_pose_stack=best_pose_stack,
        best_energies=finalization.dG[:, 0],
        # PRE-refine grid-GA genes: they do NOT parameterize the Cartesian-refined
        # best_pose_stack (see run_docking docstring). Kept as GA provenance.
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
    "run_ga_docking",
    "run_docking",
    "select_pocket_residue_blocks",
    "update_pool",
]
