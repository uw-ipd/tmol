import attr
import torch

from tmol.types.torch import Tensor
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes


@attr.s(auto_attribs=True)
class PoseStack:

    packed_block_types: PackedBlockTypes
    # residues: List[List[Residue]]

    # residue_coords: NDArray[numpy.float32][:, :, :, 3]

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

    device: torch.device

    #################### PROPERTIES #####################

    def __len__(self):
        """return the number of PoseStack held in this stack"""
        return self.coords.shape[0]

    @property
    def n_poses(self):
        return self.coords.shape[0]

    @property
    def max_n_blocks(self):
        return self.block_coord_offset.shape[1]

    @property
    def max_n_atoms(self):
        return self.packed_block_types.max_n_atoms

    @property
    def max_n_block_atoms(self):
        """Same thing as max_n_atoms"""
        return self.packed_block_types.max_n_atoms

    @property
    def max_n_pose_atoms(self):
        """The largest number of atoms in any pose"""
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
    def real_atoms(self):
        """return the boolean vector of the real atoms in the coords tensor"""
        # get the list of real atoms to read out of pose coords
        n_ats_per_pose_arange_expanded = (
            torch.arange(self.max_n_pose_atoms, dtype=torch.int64, device=self.device)
            .repeat(self.n_poses)
            .view(self.n_poses, self.max_n_pose_atoms)
        )
        n_ats_per_pose = torch.sum(self.n_ats_per_block, dim=1).unsqueeze(1)
        return n_ats_per_pose_arange_expanded < n_ats_per_pose

    def expand_coords(self):
        """Load the coordinates into a 4D tensor:
        n_poses x max_n_blocks x max_n_atoms_per_block x 3
        making it possible to perform simple operations on the
        per-block level in python/torch
        """

        # get the list of real atoms that we will be writing to in the 4D tensor
        n_ats_per_block_arange_expanded = (
            torch.arange(self.max_n_block_atoms, dtype=torch.int64, device=self.device)
            .repeat(self.n_poses * self.max_n_blocks)
            .view(self.n_poses, self.max_n_blocks, self.max_n_block_atoms)
        )

        # n_ats_per_block = self.n_ats_per_block.to(torch.int64)
        real_expanded_pose_ats = (
            n_ats_per_block_arange_expanded < self.n_ats_per_block.unsqueeze(2)
        )

        # now perform the actual copy
        expanded_coords = torch.zeros(
            (self.n_poses, self.max_n_blocks, self.max_n_block_atoms, 3),
            dtype=torch.float32,
            device=self.device,
        )
        expanded_coords[real_expanded_pose_ats] = self.coords[self.real_atoms]
        return expanded_coords, real_expanded_pose_ats

    @property
    def n_res_per_pose(self):
        return torch.sum(self.block_type_ind >= 0, dim=1)

    def is_real_block(self, pose_ind: int, block_ind: int) -> bool:
        """Report whether a particular block on a particular pose is
        real or just filler
        """
        return self.block_type_ind[pose_ind, block_ind] >= 0

    def block_type(self, pose_ind: int, block_ind: int) -> RefinedResidueType:
        """Look up the block type for a particular pose and block and retrieve it
        from the PackedBlockTypes object. is_real_block must return True"""
        return self.packed_block_types.active_block_types[
            self.block_type_ind[pose_ind, block_ind]
        ]
