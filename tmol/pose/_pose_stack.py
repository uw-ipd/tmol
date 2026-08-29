from typing import TYPE_CHECKING

import attr
import torch

from tmol.types import Tensor
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PDBInfo,
    PackedBlockTypes,
    ConstraintSet,
)

if TYPE_CHECKING:
    from tmol.pose._split_block_mapping import SplitBlockMapping


@attr.s(auto_attribs=True)
class PoseStack:
    """Batch of molecular systems with shared residue-type definitions.

    Chemistry and connectivity are fixed after construction, while ``coords``
    may be updated in place during minimization. Use :func:`attr.evolve` for
    structural metadata changes and replace or clone ``coords`` to avoid sharing
    mutable coordinate storage between pose stacks.

    Args:
        packed_block_types: Residue types and their score-term annotations.
        coords: Cartesian coordinates shaped ``[pose, atom, xyz]``.
        block_coord_offset: Per-residue atom offsets shaped ``[pose, residue]``.
        block_coord_offset64: 64-bit copy of ``block_coord_offset`` for PyTorch.
        inter_residue_connections: Connected residue and connection indices.
        inter_residue_connections64: 64-bit connection-index copy.
        inter_block_bondsep: Capped bond separation between residue connections.
        inter_block_bondsep64: 64-bit bond-separation copy.
        block_type_ind: Packed block-type index for each residue; ``-1`` is padding.
        block_type_ind64: 64-bit block-type-index copy.
        chain_id: Chain index for each residue.
        chain_id64: 64-bit chain-index copy.
        pdb_info: Source labels, occupancy, and B-factor metadata.
        constraint_set: Optional geometric constraints.
        device: Device holding all pose tensors.
        split_block_mapping: Optional mapping back to pre-split residue blocks.
    """

    packed_block_types: PackedBlockTypes

    # coordinates are held as [n-poses x max-n-atoms x 3]
    # where the offset for each residue are held in the
    # block_coord_offset tensor [n-poses x max-n-blocks]
    coords: Tensor[torch.float32][:, :, 3]

    block_coord_offset: Tensor[torch.int32][:, :]
    block_coord_offset64: Tensor[torch.int64][:, :]

    inter_residue_connections: Tensor[torch.int32][:, :, :, 2]
    inter_residue_connections64: Tensor[torch.int64][:, :, :, 2]

    inter_block_bondsep: Tensor[torch.int32][:, :, :, :, :]
    inter_block_bondsep64: Tensor[torch.int64][:, :, :, :, :]

    block_type_ind: Tensor[torch.int32][:, :]
    block_type_ind64: Tensor[torch.int64][:, :]

    chain_id: Tensor[torch.int32][:, :]
    chain_id64: Tensor[torch.int64][:, :]

    pdb_info: PDBInfo
    constraint_set: ConstraintSet | None

    device: torch.device
    split_block_mapping: "SplitBlockMapping | None" = None

    #################### INIT #####################

    def __attrs_post_init__(self) -> None:
        n_poses = self.block_coord_offset.size(0)
        n_blocks = self.block_coord_offset.size(1)

        block_inds = torch.zeros_like(self.block_coord_offset)
        block_inds[:, :] = torch.arange(0, n_blocks)
        self.block_ind_for_rot = block_inds.flatten()

        pose_inds = (
            torch.arange(0, n_poses, dtype=torch.int32, device=self.device)
            .unsqueeze(1)
            .expand((n_poses, n_blocks))
        )
        self.pose_ind_for_rot = pose_inds.flatten()

        self.block_type_ind_for_rot = self.block_type_ind.flatten()

        self.rot_offset_for_block = torch.arange(
            0, n_poses * n_blocks, dtype=torch.int32, device=self.device
        ).view(n_poses, n_blocks)
        self.first_rot_for_block = self.rot_offset_for_block
        self.first_rot_block_type = self.block_type_ind

        self.n_rots_for_pose = torch.tensor(
            [n_blocks], dtype=torch.int32, device=self.device
        ).expand(n_poses)
        self.rot_offset_for_pose = self.n_rots_for_pose * torch.arange(
            0, n_poses, dtype=torch.int32, device=self.device
        )
        coord_offset_for_pose = self.coords.size(1) * torch.arange(
            0, n_poses, dtype=torch.int32, device=self.device
        )
        self.n_rots_for_block = torch.full_like(self.block_coord_offset, 1)

        self.rot_coord_offset = (
            self.block_coord_offset.flatten()
            + torch.repeat_interleave(coord_offset_for_pose, n_blocks)
        )

        self.max_n_rots_per_pose = n_blocks

        pose_atom_offsets = self.rot_coord_offset.index_select(
            0, self.rot_offset_for_pose
        )
        atom_to_pose = torch.zeros(
            self.coords.size(0) * self.coords.size(1),
            dtype=torch.int32,
            device=self.device,
        )
        atom_to_pose[pose_atom_offsets] = 1
        atom_to_pose[0] = 0
        self.pose_ind_for_atom = atom_to_pose.cumsum(0, dtype=torch.int32)

    #################### PROPERTIES #####################

    def __len__(self) -> int:
        """Return the number of poses in this stack."""
        return self.coords.shape[0]

    @property
    def n_poses(self) -> int:
        """Return the number of poses in the stack."""
        return self.coords.shape[0]

    @property
    def max_n_blocks(self) -> int:
        """Return the padded residue count per pose."""
        return self.block_coord_offset.shape[1]

    @property
    def max_n_atoms(self) -> int:
        """Return the maximum atom count among packed residue types."""
        return self.packed_block_types.max_n_atoms

    @property
    def max_n_block_atoms(self) -> int:
        """Return the maximum atoms in any packed residue type."""
        return self.packed_block_types.max_n_atoms

    @property
    def max_n_pose_atoms(self) -> int:
        """Return the padded atom dimension of each pose."""
        return self.coords.shape[1]

    @property
    def n_ats_per_block(self) -> Tensor[torch.int64][:, :]:
        """Return the number of atoms in each block"""

        n_ats_per_block = torch.zeros(
            (self.n_poses, self.max_n_blocks), dtype=torch.int64, device=self.device
        )
        n_ats_per_block[self.block_type_ind != -1] = self.packed_block_types.n_atoms[
            self.block_type_ind[self.block_type_ind != -1].to(torch.int64)
        ].to(torch.int64)
        return n_ats_per_block

    @property
    def real_atoms(self) -> Tensor[torch.bool][:, :]:
        """Return the mask of real, non-padding atoms in ``coords``."""
        # get the list of real atoms to read out of pose coords
        n_ats_per_pose_arange_expanded = (
            torch.arange(self.max_n_pose_atoms, dtype=torch.int64, device=self.device)
            .repeat(self.n_poses)
            .view(self.n_poses, self.max_n_pose_atoms)
        )
        n_ats_per_pose = torch.sum(self.n_ats_per_block, dim=1).unsqueeze(1)
        return n_ats_per_pose_arange_expanded < n_ats_per_pose

    def clone(self) -> "PoseStack":
        """Deep-copy clone of this PoseStack"""
        new_constraint_set = (
            self.constraint_set.clone() if self.constraint_set is not None else None
        )
        return PoseStack(
            packed_block_types=self.packed_block_types,
            coords=self.coords.detach().clone(),
            block_coord_offset=self.block_coord_offset.detach().clone(),
            block_coord_offset64=self.block_coord_offset64.detach().clone(),
            inter_residue_connections=self.inter_residue_connections.detach().clone(),
            inter_residue_connections64=self.inter_residue_connections64.detach().clone(),
            inter_block_bondsep=self.inter_block_bondsep.detach().clone(),
            inter_block_bondsep64=self.inter_block_bondsep64.detach().clone(),
            block_type_ind=self.block_type_ind.detach().clone(),
            block_type_ind64=self.block_type_ind64.detach().clone(),
            chain_id=self.chain_id.detach().clone(),
            chain_id64=self.chain_id64.detach().clone(),
            pdb_info=self.pdb_info,
            constraint_set=new_constraint_set,
            device=self.device,
            split_block_mapping=self.split_block_mapping,
        )

    def split(self, index: int) -> "PoseStack":
        """Copy one pose into a new single-pose stack."""
        return PoseStack(
            packed_block_types=self.packed_block_types,
            coords=self.coords[index : index + 1].detach().clone(),
            block_coord_offset=self.block_coord_offset[index : index + 1]
            .detach()
            .clone(),
            block_coord_offset64=self.block_coord_offset64[index : index + 1]
            .detach()
            .clone(),
            inter_residue_connections=self.inter_residue_connections[index : index + 1]
            .detach()
            .clone(),
            inter_residue_connections64=self.inter_residue_connections64[
                index : index + 1
            ]
            .detach()
            .clone(),
            inter_block_bondsep=self.inter_block_bondsep[index : index + 1]
            .detach()
            .clone(),
            inter_block_bondsep64=self.inter_block_bondsep64[index : index + 1]
            .detach()
            .clone(),
            block_type_ind=self.block_type_ind[index : index + 1].detach().clone(),
            block_type_ind64=self.block_type_ind64[index : index + 1].detach().clone(),
            chain_id=self.chain_id[index : index + 1].detach().clone(),
            chain_id64=self.chain_id64[index : index + 1].detach().clone(),
            pdb_info=self.pdb_info.split(index),
            constraint_set=(
                None
                if self.constraint_set is None
                else self.constraint_set.split(index)
            ),
            device=self.device,
            split_block_mapping=(
                None
                if self.split_block_mapping is None
                else self.split_block_mapping.split(index)
            ),
        )

    def expand_coords(
        self,
    ) -> tuple[Tensor[torch.float32][:, :, :, 3], Tensor[torch.bool][:, :, :]]:
        """Expand packed coordinates into residue-major layout.

        Returns:
            Coordinates shaped ``[pose, residue, residue_atom, xyz]`` and the
            corresponding real-atom mask shaped ``[pose, residue, residue_atom]``.
        """

        # get the list of real atoms that we will be writing to in the 4D tensor
        n_ats_per_block_arange_expanded = (
            torch.arange(self.max_n_block_atoms, dtype=torch.int64, device=self.device)
            .repeat(self.n_poses * self.max_n_blocks)
            .view(self.n_poses, self.max_n_blocks, self.max_n_block_atoms)
        )
        real_expanded_pose_ats = (
            n_ats_per_block_arange_expanded < self.n_ats_per_block.unsqueeze(2)
        )

        # now perform the actual copy
        expanded_coords = torch.zeros(
            (self.n_poses, self.max_n_blocks, self.max_n_block_atoms, 3),
            dtype=self.coords.dtype,
            device=self.device,
        )
        expanded_coords[real_expanded_pose_ats] = self.coords[self.real_atoms]
        return expanded_coords, real_expanded_pose_ats

    @property
    def n_res_per_pose(self) -> Tensor[torch.int64][:]:
        """Return the number of real residues in each pose."""
        return torch.sum(self.block_type_ind >= 0, dim=1)

    def is_real_block(self, pose_ind: int, block_ind: int) -> torch.Tensor:
        """Return a scalar boolean tensor indicating whether a block is real."""
        return self.block_type_ind[pose_ind, block_ind] >= 0

    def block_type(self, pose_ind: int, block_ind: int) -> RefinedResidueType:
        """Look up the block type for a particular pose and block and retrieve it
        from the PackedBlockTypes object. is_real_block must return True"""
        return self.packed_block_types.active_block_types[
            self.block_type_ind[pose_ind, block_ind]
        ]

    def get_constraint_set(self) -> ConstraintSet | None:
        """Return the optional constraint set associated with this pose stack."""
        return self.constraint_set

    def block_identity_map(self) -> Tensor[torch.int32][:, :]:
        """Return each residue's padded block index for every pose."""
        identity_map = torch.zeros_like(self.block_coord_offset)
        identity_map[:, :] = torch.arange(
            self.block_coord_offset.size(1), device=self.device
        )
        return identity_map
