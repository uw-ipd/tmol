import torch
import numpy

from tmol.types.torch import Tensor
from tmol.pose.pdb_info import PDBInfo, DEFAULT_ATOM_B_FACTOR, DEFAULT_ATOM_OCCUPANCY
from tmol.pose.pose_stack import PoseStack
from tmol.pack.rotamer.build_rotamers import RotamerSet
from tmol.utility.cumsum import exclusive_cumsum2d_w_totals


def impose_top_rotamer_assignments(
    orig_pose_stack: PoseStack,
    rotamer_set: RotamerSet,
    assignment: Tensor[torch.int32][:, :, :],
):
    """Impose the lowest-energy rotamer assignemnt to each pose in the original PoseStack."""

    # Going through PoseStack's data members; what will be new, what will be unchanged
    #
    # -- packed_block_types: unchanged
    #
    # -- coords: new -- the whole point!
    #
    # -- block_coords_offsets: has to be updated because the number of atoms per residue may have
    #    changed
    #
    # -- inter_residue_connections: unchanged; the packer cannot change inter-block connections
    #
    # -- inter_block_bondsep: unchanged
    #
    # -- block_type_ind: new as the block types may have changed

    pbt = orig_pose_stack.packed_block_types
    device = orig_pose_stack.device
    # Ensure assignment is on the same device as the pose stack
    # (may differ when interaction graph ran on CPU for MPS).
    assignment = assignment.to(device)
    n_poses = orig_pose_stack.n_poses
    max_n_blocks = orig_pose_stack.max_n_blocks
    max_n_atoms_per_block = orig_pose_stack.max_n_atoms

    # lets figure out how many atoms per pose

    new_block_type_ind64 = torch.full(
        (n_poses, max_n_blocks), -1, dtype=torch.int64, device=device
    )
    new_rot_for_block64 = (
        assignment[:, 0, :].to(torch.int64) + rotamer_set.rot_offset_for_block
    )

    is_real_block = orig_pose_stack.block_type_ind64 != -1

    new_block_type_ind64[is_real_block] = rotamer_set.block_type_ind_for_rot[
        new_rot_for_block64[is_real_block]
    ]
    blocks_with_changed_types = new_block_type_ind64 != orig_pose_stack.block_type_ind64
    any_block_types_changed = torch.any(blocks_with_changed_types).item()

    new_n_atoms_per_block32 = torch.zeros(
        (n_poses, max_n_blocks), dtype=torch.int32, device=device
    )
    new_n_atoms_per_block32[is_real_block] = pbt.n_atoms[
        new_block_type_ind64[is_real_block]
    ]
    new_n_atoms_per_block64 = new_n_atoms_per_block32.to(torch.int64)

    if any_block_types_changed:
        # get the per-pose offset for each block w/ exclusive cumsum on n-atoms-per-block
        new_n_atoms_offset32, new_n_pose_atoms = exclusive_cumsum2d_w_totals(
            new_n_atoms_per_block32
        )
        new_n_atoms_offset64 = new_n_atoms_offset32.to(torch.int64)
        new_max_n_pose_atoms = int(torch.max(new_n_pose_atoms).item())
    else:
        new_n_atoms_offset32 = orig_pose_stack.block_coord_offset
        new_n_atoms_offset64 = orig_pose_stack.block_coord_offset64
        new_max_n_pose_atoms = orig_pose_stack.max_n_pose_atoms

    # okay, now lets preprare the indices for our copy operation
    # let's think about it like this: we have a 3D tensor with i, j, k indices representing
    # pose-ind, block-ind, and atom-ind.
    # For the dst indices, we add a per-pose offset i * new_max_n_pose_atoms
    # and we add a block-offset from new_n_atoms_offset64[j].
    # For the src indices, we take the rotamer assigned to pose-i-residue-j,
    # and that rotamer gives us the offset into the rotamer_set.coords tensor.
    max_n_atoms_arange64 = torch.arange(
        max_n_atoms_per_block, dtype=torch.int64, device=device
    )
    max_n_atoms_arange64 = max_n_atoms_arange64.view(1, 1, -1).expand(
        n_poses, max_n_blocks, max_n_atoms_per_block
    )

    pose_for_atom64 = torch.arange(n_poses, dtype=torch.int64, device=device)
    pose_for_atom64 = pose_for_atom64.view(-1, 1, 1).expand(
        n_poses, max_n_blocks, max_n_atoms_per_block
    )

    pose_offset_for_atom64 = (
        torch.arange(n_poses, dtype=torch.int64, device=device) * new_max_n_pose_atoms
    )
    pose_offset_for_atom64 = pose_offset_for_atom64.view(-1, 1, 1).expand(
        n_poses, max_n_blocks, max_n_atoms_per_block
    )

    block_for_atom64 = (
        torch.arange(max_n_blocks, dtype=torch.int64, device=device)
        .view(1, -1, 1)
        .expand(n_poses, max_n_blocks, max_n_atoms_per_block)
    )

    pose_coords1d_offset_for_atom64 = (
        new_n_atoms_offset64[pose_for_atom64, block_for_atom64] + pose_offset_for_atom64
    )

    new_n_atoms_for_atoms_block64 = new_n_atoms_per_block64.unsqueeze(2).expand(
        n_poses, max_n_blocks, max_n_atoms_per_block
    )
    is_pose_atom_real = max_n_atoms_arange64 < new_n_atoms_for_atoms_block64

    dst_inds = (pose_coords1d_offset_for_atom64 + max_n_atoms_arange64)[
        is_pose_atom_real
    ]

    rot_coord_offset_for_block32 = torch.full(
        (n_poses, max_n_blocks), -1, dtype=torch.int32, device=device
    )
    rot_coord_offset_for_block32[is_real_block] = rotamer_set.coord_offset_for_rot[
        new_rot_for_block64[is_real_block]
    ]
    rot_coord_offset_for_block64 = rot_coord_offset_for_block32.to(torch.int64)
    rot_coord_offset_for_atom64 = rot_coord_offset_for_block64.unsqueeze(2).expand(
        n_poses, max_n_blocks, max_n_atoms_per_block
    )
    src_inds = (rot_coord_offset_for_atom64 + max_n_atoms_arange64)[is_pose_atom_real]

    # now lets copy the coordinates
    new_coords = torch.zeros(
        (n_poses * new_max_n_pose_atoms, 3), dtype=torch.float32, device=device
    )
    new_coords[dst_inds] = rotamer_set.coords[src_inds]
    new_coords = new_coords.view(n_poses, new_max_n_pose_atoms, 3)

    # Now let's create a new PDBInfo object for the new PoseStack
    # We can mostly copy from the original PoseStack's PDBInfo,
    # but where block-types have changed, we have to be careful with
    # the occupancies and b-factors
    if any_block_types_changed:
        # need to make new atom_occupancy and atom_b_factor arrays
        orig_pdb_info = orig_pose_stack.pdb_info
        new_atom_occupancy = numpy.zeros(
            (n_poses, new_max_n_pose_atoms), dtype=numpy.float32
        )
        new_atom_b_factor = numpy.zeros(
            (n_poses, new_max_n_pose_atoms), dtype=numpy.float32
        )
        _, real_expanded_pose_ats = orig_pose_stack.expand_coords()
        real_expanded_pose_ats = real_expanded_pose_ats.cpu().numpy()
        orig_real_atoms = orig_pose_stack.real_atoms.cpu().numpy()
        expanded_b_factor = numpy.full(
            (n_poses, max_n_blocks, pbt.max_n_atoms),
            DEFAULT_ATOM_B_FACTOR,
            dtype=numpy.float32,
        )
        expanded_occupancy = numpy.full(
            (n_poses, max_n_blocks, pbt.max_n_atoms),
            DEFAULT_ATOM_OCCUPANCY,
            dtype=numpy.float32,
        )
        new_expanded_b_factor = numpy.full(
            (n_poses, max_n_blocks, pbt.max_n_atoms),
            DEFAULT_ATOM_B_FACTOR,
            dtype=numpy.float32,
        )
        new_expanded_occupancy = numpy.full(
            (n_poses, max_n_blocks, pbt.max_n_atoms),
            DEFAULT_ATOM_OCCUPANCY,
            dtype=numpy.float32,
        )
        # keep all the b-factors for blocks that did not change block type
        blocks_w_unchanged_types = (
            torch.logical_not(blocks_with_changed_types).cpu().numpy()
        )
        expanded_b_factor[real_expanded_pose_ats] = orig_pdb_info.atom_b_factor[
            orig_real_atoms
        ]
        expanded_occupancy[real_expanded_pose_ats] = orig_pdb_info.atom_occupancy[
            orig_real_atoms
        ]
        new_expanded_b_factor[blocks_w_unchanged_types] = expanded_b_factor[
            blocks_w_unchanged_types
        ]
        new_expanded_occupancy[blocks_w_unchanged_types] = expanded_occupancy[
            blocks_w_unchanged_types
        ]
        # TO DO: now handle blocks that did change type?
        # PUNT!

        is_real_atom_new = torch.zeros(
            (n_poses, max_n_blocks, pbt.max_n_atoms),
            dtype=torch.bool,
            device=pbt.device,
        )
        is_real_atom_new[is_real_block] = pbt.atom_is_real[
            new_block_type_ind64[is_real_block]
        ]
        nz_is_real_pose_ind_new, nz_is_real_block_ind_new, nz_is_real_atom_ind_new = (
            torch.nonzero(is_real_atom_new, as_tuple=True)
        )
        # new_atom_b_factor[is_real_atom]
        new_pose_at_is_real = torch.arange(
            new_max_n_pose_atoms, dtype=torch.int64, device=device
        ).repeat(n_poses, 1) < new_n_pose_atoms.unsqueeze(1)
        new_pose_at_is_real = new_pose_at_is_real.cpu().numpy()

        new_atom_b_factor[new_pose_at_is_real] = new_expanded_b_factor[
            nz_is_real_pose_ind_new.cpu().numpy(),
            nz_is_real_block_ind_new.cpu().numpy(),
            nz_is_real_atom_ind_new.cpu().numpy(),
        ]
        new_atom_occupancy[new_pose_at_is_real] = new_expanded_occupancy[
            nz_is_real_pose_ind_new.cpu().numpy(),
            nz_is_real_block_ind_new.cpu().numpy(),
            nz_is_real_atom_ind_new.cpu().numpy(),
        ]
        new_pdb_info = PDBInfo(
            residue_labels=orig_pose_stack.pdb_info.residue_labels,
            residue_insertion_codes=orig_pose_stack.pdb_info.residue_insertion_codes,
            chain_labels=orig_pose_stack.pdb_info.chain_labels,
            atom_occupancy=new_atom_occupancy,
            atom_b_factor=new_atom_b_factor,
        )
    else:
        new_pdb_info = orig_pose_stack.pdb_info

    # now construct the new PoseStack
    new_pose_stack = PoseStack(
        packed_block_types=pbt,
        coords=new_coords,
        block_coord_offset=new_n_atoms_offset32,
        block_coord_offset64=new_n_atoms_offset64,
        inter_residue_connections=orig_pose_stack.inter_residue_connections,
        inter_residue_connections64=orig_pose_stack.inter_residue_connections64,
        inter_block_bondsep=orig_pose_stack.inter_block_bondsep,
        inter_block_bondsep64=orig_pose_stack.inter_block_bondsep64,
        block_type_ind=new_block_type_ind64.to(torch.int32),
        block_type_ind64=new_block_type_ind64,
        chain_id=orig_pose_stack.chain_id,
        chain_id64=orig_pose_stack.chain_id64,
        pdb_info=new_pdb_info,
        constraint_set=orig_pose_stack.constraint_set,
        device=device,
    )
    return new_pose_stack
