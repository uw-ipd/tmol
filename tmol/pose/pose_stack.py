import attr
import torch

from tmol.types.torch import Tensor
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.constraint_set import ConstraintSet


@attr.s(auto_attribs=True)
class PoseStack:
    """The PoseStack class defines a batch (a stack) of molecular systems

    The PoseStack defines the per-residue chemistry, inter-residue
    connectivity, and coordinates of a set of molecular systems.
    The per-residue chemistry and the connectivity are meant to be
    constant over its lifetime; however, its coordinates are allowed
    to change. That is, a PoseStack may have its coords tensor written
    to without any negative consequences. Thus, you can minimize the
    coordinates in a PoseStack.

    Datamembers:
    packed_block_types: a representation of the chemical space that this
    PoseStack can contain. The PackedBlockTypes object aggregates a set
    of residue-types objects (RefinedResidueTypes) and holds annotations
    for this aggregate that must be made by the terms in the ScoreFunction
    in order for them to efficiently perform their calculations.

    coords: a tensor of [n_poses x max_n_atoms_per_pose x 3] holding the
    cartesian coordinates of the atoms in the system. The coordinates
    of the atoms are held in a contiguous array so that mixing very
    large residue types (e.g. heme) and very small residue types
    (e.g. water) does not waste memory / GPU cache.

    block_coord_offset: a tensor of [n_poses x max_n_residues] holding
    the starting indices in the coords tensor for the residues; offsets
    for custom kernels are 32-bit integers, offset for torch functions
    are 64-bit integers. We keep around both for performance reasons.

    inter_residue_connections: a tensor of
    [n_poses x max_n_residues x max_n_conn x 2] representing for each
    inter-residue connection point on each residue the 1) index of
    the residue it is connected to (sentinel of -1 for "no connection
    defined) and the connection-point index it is connected to
    (sentinel of -1, also).

    inter_block_bondsep: a integer tensor of shape
    [n_poses x max_n_residues x max_n_residues x max_n_conn x max_n_conn]
    stating the number of chemical bonds that separate every pair of
    inter-residue connections for every pair of residues -- up to a
    maximum inter-residue separation of
    tmol.chemical.MAX_SIG_BOND_SEPARATION (6 as of March 2024) --
    so that the number of chemical bonds separating arbitrary
    atom pairs may be rapidly computed for the interatomic energy
    calculations

    block_type_ind: the integer index for each block type (residue type)
    referring to the order in which that block type appears in the
    PoseStack's PackedBlockTypes object. A sentinel of -1 for positions
    where there is no block type.

    device: the torch.device that this collection of structures lives on
    """

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

    #################### INIT #####################

    def __attrs_post_init__(self):

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

        # self.rot_coord_offset = _p(rotamer_set.rot_coord_offset)
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

        # pose_coord_offset = torch.cumsum(self.n_rots_for_pose, 0).roll(1,0)

        # print("rot_offset_for_pose: ", self.rot_offset_for_pose)
        # print("pose_coord_offset: ", pose_coord_offset)
        self.rot_coord_offset = (
            self.block_coord_offset.flatten()
            + torch.repeat_interleave(coord_offset_for_pose, n_blocks)
        )
        # print("rot_coord_offset: ", self.rot_coord_offset)
        # print("block_coord_offset: ", self.block_coord_offset)

        self.max_n_rots_per_pose = n_blocks

        pose_atom_offsets = self.rot_coord_offset.index_select(
            0, self.rot_offset_for_pose
        )
        atom_to_pose = torch.zeros(
            self.coords.size(0) * self.coords.size(1),
            dtype=torch.int32,
            device=self.device,
        )
        # print(self.coords.size(0) * self.coords.size(1), self.n_poses, n_poses)
        atom_to_pose[pose_atom_offsets] = 1
        atom_to_pose[0] = 0
        self.pose_ind_for_atom = atom_to_pose.cumsum(0, dtype=torch.int32)

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

    def get_constraint_set(self):
        # make a constraint set if it doesn't exist
        if not hasattr(self, "_constraint_set"):
            self._constraint_set = ConstraintSet(device=self.device)

        # ensure the constraint set points back at us (after creation or deep copy)
        self._constraint_set.pose_stack = self
        return self._constraint_set

    def block_identity_map(self):
        # print("bco size: ", self.block_coord_offset.size())
        identity_map = torch.zeros_like(self.block_coord_offset)
        # print("im size: ", identity_map.size())
        identity_map[:, :] = torch.arange(
            self.block_coord_offset.size(1), device=self.device
        )
        return identity_map
