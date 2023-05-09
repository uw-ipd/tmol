import torch
import numpy

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.datatypes import KinForest
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.check_fold_forest import mark_polymeric_bonds_in_foldforest_edges


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


@validate_args
def get_all_intrablock_bonds(pose_stack: PoseStack) -> Tensor[torch.int32][:, 2]:
    pbt = pose_stack.packed_block_types
    device = pose_stack.device

    real_blocks = pose_stack.block_type_ind != -1
    real_bonds = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_bonds),
        dtype=torch.bool,
        device=device,
    )
    real_bonds[real_blocks] = pbt.bond_is_real[pose_stack.block_type_ind64[real_blocks]]
    intrablock_bonds = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_bonds, 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    intrablock_bonds[real_blocks] = pbt.bond_indices[
        pose_stack.block_type_ind64[real_blocks]
    ]
    nz_real_bond_pose_ind, nz_real_bond_block_ind, _ = torch.nonzero(
        real_bonds, as_tuple=True
    )

    intrablock_bonds = intrablock_bonds[real_bonds]
    intrablock_bonds += (
        pose_stack.max_n_pose_atoms * nz_real_bond_pose_ind
        + pose_stack.block_coord_offset[nz_real_bond_pose_ind, nz_real_bond_block_ind]
    ).unsqueeze(1)
    return intrablock_bonds


def get_atom_inds_for_interblock_connections(
    pose_stack: PoseStack,
    real_blocks: Tensor[torch.bool][:, :],
    nz_real_block_pose_ind_prelim: Tensor[torch.int64],
    nz_real_block_block_ind_prelim: Tensor[torch.int64],
    src_connections: Tensor[torch.int64][:, :],
    dst_connections: Tensor[torch.int64][:, :],
    kinematic_connections: Tensor[torch.bool][:, :, :],
):
    """Find the atoms indices for bonds that should be included in the KinForest
    of a certain type. The src_connections tensor holds the inter-block connection
    indices of one type that are present in the PoseStack, the dst_connections tensor
    holds the inter-block connection indices of a second type, and the
    kinematic_connections tensor at position [p, b1, b2] is a yes or no as
    to whether a src-to-dst connection is desired in the KinForest between block
    b1 and block b2 in pose p. In particular, this is useful for figuring
    out whether the polymeric connections in a pose should be included in its
    fold tree; the logic for handling up-to-down connections (i.e. N->C) is identical
    to the logic for handling down-to-up connections (i.e. C->N).
    """

    pbt = pose_stack.packed_block_types

    # find the index of the other block that the src block is connected to;
    # not all connections are complete, so consider this a "preliminary" set
    # that we will refine into the set of complete connections
    src_conn_other_block_prelim = pose_stack.inter_residue_connections64[
        nz_real_block_pose_ind_prelim,
        nz_real_block_block_ind_prelim,
        src_connections[real_blocks],
        0,
    ]
    src_conn_other_conn_prelim = pose_stack.inter_residue_connections64[
        nz_real_block_pose_ind_prelim,
        nz_real_block_block_ind_prelim,
        src_connections[real_blocks],
        1,
    ]

    # find which of the connections are complete: as in, there's another residue
    # on the other side of the connection point and, having found the complete
    # connections, go back and refine the list of pose-inds and block-inds that
    # we will work with
    src_conn_complete = src_conn_other_block_prelim != -1

    src_conn_other_block = src_conn_other_block_prelim[src_conn_complete]
    src_conn_other_conn = src_conn_other_conn_prelim[src_conn_complete]
    nz_real_block_pose_ind_src = nz_real_block_pose_ind_prelim[src_conn_complete]
    nz_real_block_block_ind_src = nz_real_block_block_ind_prelim[src_conn_complete]

    # how do I tell if the src-conn on residue x is connected to the
    # dst conn on residue y? if the index of the input dst-connection
    # is the same as the one that the src-connection is connected to

    connection_is_src_to_dst = (
        dst_connections[nz_real_block_pose_ind_src, src_conn_other_block]
        == src_conn_other_conn
    )

    # now lets refine the set of indices we have constructed so far:
    # not only are we looking at the set of complete connections from the
    # src but we are now looking at the set that actually meet the dst
    # conns
    src_polyconn_pose_ind = nz_real_block_pose_ind_src[connection_is_src_to_dst]
    src_polyconn_block_ind = nz_real_block_block_ind_src[connection_is_src_to_dst]
    src_polyconn_other_block_ind = src_conn_other_block[connection_is_src_to_dst]
    src_polyconn_conn_ind = src_connections[real_blocks][src_conn_complete][
        connection_is_src_to_dst
    ]
    src_polyconn_other_conn_ind = src_conn_other_conn[connection_is_src_to_dst]

    # Now we ask: are the src-to-dst connections that we have identified that exist
    # in the PoseStack actually desired as part of the kinematic forest? The
    # kinematic_connections tensor will tell us when we index into it using
    # [pose, src-block, dst-block] indices. Note that sometimes a src-to-dst
    # connection is not desired but instead the dst-to-src connection is. The
    # kinematic_connections tensor is not symmetric.
    src_conn_in_fold_forest = kinematic_connections[
        src_polyconn_pose_ind, src_polyconn_block_ind, src_polyconn_other_block_ind
    ]

    # ok, now refine our index tensors to the kinematically desired
    src_kin_pose_ind = src_polyconn_pose_ind[src_conn_in_fold_forest]
    src_kin_block_ind = src_polyconn_block_ind[src_conn_in_fold_forest]
    src_kin_other_block_ind = src_polyconn_other_block_ind[src_conn_in_fold_forest]
    src_kin_conn_ind = src_polyconn_conn_ind[src_conn_in_fold_forest]
    src_kin_other_conn_ind = src_polyconn_other_conn_ind[src_conn_in_fold_forest]

    # finally, we will compute the global indices of the atoms that form the
    # bonds that we desire as ordered pairs: the first atom being kinematically
    # upstream of the second atom
    src_kin_bond_inds_1 = (
        pose_stack.max_n_pose_atoms * src_kin_pose_ind
        + pose_stack.block_coord_offset[src_kin_pose_ind, src_kin_block_ind]
        + pbt.conn_atom[
            pose_stack.block_type_ind64[src_kin_pose_ind, src_kin_block_ind],
            src_kin_conn_ind,
        ]
    )
    src_kin_bond_inds_2 = (
        pose_stack.max_n_pose_atoms * src_kin_pose_ind
        + pose_stack.block_coord_offset[src_kin_pose_ind, src_kin_other_block_ind]
        + pbt.conn_atom[
            pose_stack.block_type_ind64[src_kin_pose_ind, src_kin_other_block_ind],
            src_kin_other_conn_ind,
        ]
    )
    src_kin_bond_inds = torch.stack((src_kin_bond_inds_1, src_kin_bond_inds_2), dim=1)
    return src_kin_bond_inds


def get_polymeric_bonds_in_fold_forest(
    pose_stack: PoseStack,
    kinematic_polymeric_connections: NDArray[numpy.int64][:, :, :],
):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device
    kinematic_polymeric_connections_torch = torch.tensor(
        kinematic_polymeric_connections != 0, dtype=torch.bool, device=device
    )

    real_blocks = pose_stack.block_type_ind != -1
    down_connections = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int64,
        device=device,
    )
    down_connections[real_blocks] = pbt.down_conn_inds[
        pose_stack.block_type_ind64[real_blocks]
    ].to(torch.int64)

    up_connections = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int64,
        device=device,
    )
    up_connections[real_blocks] = pbt.up_conn_inds[
        pose_stack.block_type_ind64[real_blocks]
    ].to(torch.int64)

    nz_real_block_pose_ind_prelim, nz_real_block_block_ind_prelim = torch.nonzero(
        real_blocks, as_tuple=True
    )

    up_kin_bonds = get_atom_inds_for_interblock_connections(
        pose_stack,
        real_blocks,
        nz_real_block_pose_ind_prelim,
        nz_real_block_block_ind_prelim,
        up_connections,
        down_connections,
        kinematic_polymeric_connections_torch,
    )
    down_kin_bonds = get_atom_inds_for_interblock_connections(
        pose_stack,
        real_blocks,
        nz_real_block_pose_ind_prelim,
        nz_real_block_block_ind_prelim,
        down_connections,
        up_connections,
        kinematic_polymeric_connections_torch,
    )

    return torch.cat((up_kin_bonds, down_kin_bonds), dim=0)


def get_all_bonds(pose_stack: PoseStack):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device

    real_blocks = pose_stack.block_type_ind != -1

    # n_conn = pbt.n_conn[pose_stack.block_type_ind64[real_blocks]]
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

    intrares_bonds = get_all_intrablock_bonds(pose_stack)
    bonds = torch.cat((intrares_bonds, interres_bonds), dim=0)

    return bonds


def get_root_atom_indices(
    pose_stack: PoseStack, fold_tree_roots: NDArray[int][:]
) -> Tensor[torch.int32][:]:
    # take the first atom from each root residue
    assert pose_stack.n_poses == fold_tree_roots.shape[0]
    roots64_dev = torch.tensor(
        fold_tree_roots, dtype=torch.int64, device=pose_stack.device
    )
    n_pose_arange = torch.arange(
        pose_stack.n_poses, dtype=torch.int32, device=pose_stack.device
    )
    return (
        n_pose_arange * pose_stack.max_n_pose_atoms
        + pose_stack.block_coord_offset[n_pose_arange.to(torch.int64), roots64_dev]
    )


def construct_pose_stack_kinforest(
    pose_stack: PoseStack, fold_forest: FoldForest
) -> KinForest:
    intra_block_bonds = get_all_intrablock_bonds(pose_stack)
    kin_polymeric_connections = mark_polymeric_bonds_in_foldforest_edges(
        pose_stack.n_poses, pose_stack.max_n_blocks, fold_forest.edges
    )
    kin_polymeric_bonds = get_polymeric_bonds_in_fold_forest(
        pose_stack, kin_polymeric_connections
    )

    # TO DO: add bonds between jump atoms
    # TO DO: determine which atoms on a block a jump should
    # connect to. Logic in R3: take the central "mainchain" atom
    # which is only ok for polymers, but perverse for anything else.
    # What's the mainchain of a ligand?!
    # jump_atom_pairs = get_jump_bonds_in_fold_forest(pose_stack, fold_forest)

    all_bonds = torch.cat((intra_block_bonds, kin_polymeric_bonds), dim=0).cpu().numpy()
    tor_bonds = get_bonds_for_named_torsions(pose_stack).cpu().numpy()
    root_atoms = get_root_atom_indices(pose_stack, fold_forest.roots).cpu().numpy()

    return (
        KinematicBuilder().append_connected_components(
            root_atoms,
            *KinematicBuilder.define_trees_with_prioritized_bonds(
                roots=root_atoms, potential_bonds=all_bonds, prioritized_bonds=tor_bonds
            ),
            # to do: to_jump_nodes=jump_atom_pairs[0,:]
        )
    ).kinforest
