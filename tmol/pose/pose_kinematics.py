import torch
import numba
import numpy

from tmol.types.array import NDArray
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.datatypes import KinTree
from tmol.kinematics.fold_forest import EdgeType


def get_bonds_for_named_torsions(pose_stack: PoseStack):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device

    real_blocks = pose_stack.block_type_ind != -1

    real_tors = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_torsions),
        dtype=torch.bool,
        device=device,
    )
    real_tors[real_blocks] = pbt.torsion_is_real[
        pose_stack.block_type_ind64[real_blocks]
    ]

    per_block_tor_uaids = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_torsions, 4, 3),
        -1,
        dtype=torch.int64,
        device=device,
    )
    per_block_tor_uaids[real_blocks] = pbt.torsion_uaids[
        pose_stack.block_type_ind64[real_blocks]
    ].to(torch.int64)

    middle_bond_uaids = per_block_tor_uaids[:, :, :, 1:3, :]
    middle_bond_ats = middle_bond_uaids[:, :, :, :, 0]

    # Let's get the list of atoms that are external, identified
    # by their block index, their connection index, and number of
    # chemical bonds into the neighboring residue.
    #
    # The external atoms are the ones that 1) have a sentinel value (-1)
    # for their atom-index slot and 2) come from real torsions.
    # Unsqueeze real_tors to broadcast for both atoms that
    # define the center axis of a torsion
    at_is_external = torch.logical_and(middle_bond_ats == -1, real_tors.unsqueeze(3))

    # The internal atoms are the ones that 1) do not have a sentinel value
    # for their atom-index slot and 2) come from real torsions
    # Save this for later, we will need it.
    at_is_internal = torch.logical_and(middle_bond_ats != -1, real_tors.unsqueeze(3))

    ############################################################################
    # Let's focus on the set of external atoms for a bit
    (
        xaid_pose_ind_prelim,
        xaid_block_ind_prelim,
        xaid_tor_ind_prelim,
        xaid_at_ind_prelim,
    ) = torch.nonzero(at_is_external, as_tuple=True)

    # what is the connection index for the external atoms?
    xaid_conn_ind_prelim = middle_bond_uaids[
        xaid_pose_ind_prelim,
        xaid_block_ind_prelim,
        xaid_tor_ind_prelim,
        xaid_at_ind_prelim,
        1,
    ]
    # who is the connected partner?
    # now, in some circumstances, a connected partner is not given
    # and the pose_stack.inter_residue_connections lists the sentinel
    # value of -1. We will have to refine the set of external atoms
    # after this function returns.
    xaid_other_block_ind_prelim = pose_stack.inter_residue_connections64[
        xaid_pose_ind_prelim, xaid_block_ind_prelim, xaid_conn_ind_prelim, 0
    ]

    fully_connected = xaid_other_block_ind_prelim != -1

    xaid_pose_ind = xaid_pose_ind_prelim[fully_connected]
    xaid_block_ind = xaid_block_ind_prelim[fully_connected]
    xaid_tor_ind = xaid_tor_ind_prelim[fully_connected]
    xaid_at_ind = xaid_at_ind_prelim[fully_connected]
    xaid_conn_ind = xaid_conn_ind_prelim[fully_connected]
    xaid_other_block_ind = xaid_other_block_ind_prelim[fully_connected]

    # how many chemical bonds from the connection point is it?
    xaid_path_dist = middle_bond_uaids[
        xaid_pose_ind, xaid_block_ind, xaid_tor_ind, xaid_at_ind, 2
    ]

    xaid_other_conn_ind = pose_stack.inter_residue_connections64[
        xaid_pose_ind, xaid_block_ind, xaid_conn_ind, 1
    ]

    xaid_other_block_type = pose_stack.block_type_ind64[
        xaid_pose_ind, xaid_other_block_ind
    ]
    xaid_other_atom_ind = pbt.atom_downstream_of_conn[
        xaid_other_block_type, xaid_other_conn_ind, xaid_path_dist
    ].to(torch.int64)
    xaid_other_block_offset = pose_stack.block_coord_offset64[
        xaid_pose_ind, xaid_other_block_ind
    ]

    xaid_pose_offset = (
        pose_stack.max_n_pose_atoms
        * torch.arange(pose_stack.n_poses, dtype=torch.int64, device=device)
    )[xaid_pose_ind]

    # ok, now put the external atom ind
    middle_bond_ats[xaid_pose_ind, xaid_block_ind, xaid_tor_ind, xaid_at_ind] = (
        xaid_other_atom_ind + xaid_other_block_offset + xaid_pose_offset
    )

    #############################################################
    # Alright, now lets go back and focus on the internal atoms.
    # We need to update the atoms' indices w/ offsets

    naid_pose_ind, naid_block_ind, naid_tor_ind, naid_at_ind = torch.nonzero(
        at_is_internal, as_tuple=True
    )

    naid_pose_offset = (
        pose_stack.max_n_pose_atoms
        * torch.arange(pose_stack.n_poses, dtype=torch.int64, device=device)
    )[naid_pose_ind]

    naid_block_offset = pose_stack.block_coord_offset64[naid_pose_ind, naid_block_ind]

    middle_bond_ats[naid_pose_ind, naid_block_ind, naid_tor_ind, naid_at_ind] += (
        naid_pose_offset + naid_block_offset
    )

    real_middle_bond_ats = middle_bond_ats[real_tors]
    complete_middle_bond_ats = torch.all(real_middle_bond_ats != -1, dim=1)

    return real_middle_bond_ats[complete_middle_bond_ats]


def get_all_bonds(pose_stack: PoseStack):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device

    real_blocks = pose_stack.block_type_ind != -1
    real_bonds = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_bonds),
        dtype=torch.bool,
        device=device,
    )
    real_bonds[real_blocks] = pbt.bond_is_real[pose_stack.block_type_ind64[real_blocks]]
    intrares_bonds = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_bonds, 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    intrares_bonds[real_blocks] = pbt.bond_indices[
        pose_stack.block_type_ind64[real_blocks]
    ]
    nz_real_bond_pose_ind, nz_real_bond_block_ind, _ = torch.nonzero(
        real_bonds, as_tuple=True
    )

    intrares_bonds = intrares_bonds[real_bonds]
    intrares_bonds += (
        pose_stack.max_n_pose_atoms * nz_real_bond_pose_ind
        + pose_stack.block_coord_offset[nz_real_bond_pose_ind, nz_real_bond_block_ind]
    ).unsqueeze(1)

    n_conn = pbt.n_conn[pose_stack.block_type_ind64[real_blocks]]
    real_conn = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_conn),
        dtype=torch.bool,
        device=device,
    )
    real_conn[real_blocks] = pbt.conn_is_real[pose_stack.block_type_ind64[real_blocks]]
    (
        nz_real_conn_pose_ind_prelim,
        nz_real_conn_block_ind_prelim,
        nz_real_conn_conn_ind_prelim,
    ) = torch.nonzero(real_conn, as_tuple=True)

    other_block_prelim = pose_stack.inter_residue_connections[real_conn][:, 0]
    other_conn_prelim = pose_stack.inter_residue_connections[real_conn][:, 1]
    complete_conn = other_block_prelim != -1
    other_block = other_block_prelim[complete_conn].to(torch.int64)
    other_conn = other_conn_prelim[complete_conn].to(torch.int64)

    nz_real_conn_pose_ind = nz_real_conn_pose_ind_prelim[complete_conn]
    nz_real_conn_block_ind = nz_real_conn_block_ind_prelim[complete_conn]
    nz_real_conn_conn_ind = nz_real_conn_conn_ind_prelim[complete_conn]

    other_block_atom = pbt.conn_atom[
        pose_stack.block_type_ind64[nz_real_conn_pose_ind, other_block], other_conn
    ]

    global_at1 = (
        pose_stack.max_n_pose_atoms * nz_real_conn_pose_ind
        + pose_stack.block_coord_offset[nz_real_conn_pose_ind, nz_real_conn_block_ind]
        + pbt.conn_atom[
            pose_stack.block_type_ind64[nz_real_conn_pose_ind, nz_real_conn_block_ind],
            nz_real_conn_conn_ind,
        ]
    )
    global_at2 = (
        pose_stack.max_n_pose_atoms * nz_real_conn_pose_ind
        + pose_stack.block_coord_offset[nz_real_conn_pose_ind, other_block]
        + other_block_atom
    )

    interres_bonds = torch.stack((global_at1, global_at2), dim=1)
    bonds = torch.cat((intrares_bonds, interres_bonds), dim=0)

    return bonds


def construct_pose_stack_kintree(pose_stack: PoseStack) -> KinTree:
    kintree = (
        KinematicBuilder().append_connected_component(
            *KinematicBuilder.component_for_prioritized_bonds(
                roots=pose_stack.max_n_pose_atoms
                * numpy.arange(pose_stack.n_poses, dtype=int),
                mandatory_bonds=get_bonds_for_named_torsions(pose_stack),
                all_bonds=get_all_bonds(pose_stack),
            )
        )
    ).kintree
