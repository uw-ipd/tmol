import torch
from tmol.pose.pose_stack import PoseStack


def test_resolve_torsions(ubq_res):
    torch_device = torch.device("cpu")
    pose_stack = PoseStack.one_structure_from_polymeric_residues(
        ubq_res[:4], torch_device
    )
    pbt = pose_stack.packed_block_types

    real_blocks = pose_stack.block_type_ind != -1

    real_tors = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_torsions),
        dtype=torch.bool,
        device=torch_device,
    )
    real_tors[real_blocks] = pbt.torsion_is_real[
        pose_stack.block_type_ind64[real_blocks]
    ]

    per_block_tor_uaids = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_torsions, 4, 3),
        -1,
        dtype=torch.int64,
        device=torch_device,
    )
    per_block_tor_uaids[real_blocks] = pbt.torsion_uaids[
        pose_stack.block_type_ind64[real_blocks]
    ].to(torch.int64)
    print("per_block_tor_uaids")
    print(per_block_tor_uaids.shape)

    middle_bond_uaids = per_block_tor_uaids[:, :, :, 1:3, :]
    print("middle_bond_uaids")
    print(middle_bond_uaids.shape)
    middle_bond_ats = middle_bond_uaids[:, :, :, :, 0]

    print("middle_bond_ats")
    print(middle_bond_ats)

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

    # Let's focus entirely on the set of external atoms
    xaid_pose_ind, xaid_block_ind, xaid_tor_ind, xaid_at_ind = torch.nonzero(
        at_is_external, as_tuple=True
    )

    print("xaid_pose_ind")
    print(xaid_pose_ind)
    print("xaid_block_ind")
    print(xaid_block_ind)
    print("xaid_tor_ind")
    print(xaid_tor_ind)
    print("xaid_at_ind")
    print(xaid_at_ind)

    # what is the connection index for the external atoms?
    xaid_conn_ind = middle_bond_uaids[
        xaid_pose_ind, xaid_block_ind, xaid_tor_ind, xaid_at_ind, 1
    ]
    # how many chemical bonds from the connection point is it?
    xaid_path_dist = middle_bond_uaids[
        xaid_pose_ind, xaid_block_ind, xaid_tor_ind, xaid_at_ind, 2
    ]

    # who is the connected partner
    xaid_other_block_ind = pose_stack.inter_residue_connections64[
        xaid_pose_ind, xaid_block_ind, xaid_conn_ind, 0
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
        * torch.arange(pose_stack.n_poses, dtype=torch.int64, device=torch_device)
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
        * torch.arange(pose_stack.n_poses, dtype=torch.int64, device=torch_device)
    )[naid_pose_ind]

    naid_other_block_offset = pose_stack.block_coord_offset64[
        naid_pose_ind, naid_block_ind
    ]

    middle_bond_ats[naid_pose_ind, naid_block_ind, naid_tor_ind, naid_at_ind] += (
        naid_pose_offset + naid_other_block_offset
    )

    print(middle_bond_ats)


def test_build_kintree_for_pose(ubq_res, torch_device):
    pass
