import torch

from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms_v2,
    max_n_canonical_atoms,
)
from tmol.io.chain_deduction import chain_inds_for_pose_stack
from tmol.io.details.select_from_canonical import (
    _annotate_packed_block_types_w_canonical_res_order,
    _annotate_packed_block_types_w_dslf_conn_inds,
    _annotate_packed_block_types_w_canonical_atom_order,
)


def canonical_form_from_pose_stack(pose_stack: PoseStack, chain_id=None):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    _annotate_packed_block_types_w_canonical_atom_order(pbt)

    n_poses = pose_stack.n_poses
    max_n_res = pose_stack.max_n_blocks
    max_n_atoms_per_res = pbt.max_n_atoms

    cf_res_types = torch.full(
        (n_poses, max_n_res), -1, dtype=torch.int64, device=device
    )

    # not all blocks in the pose are real
    is_real_block = pose_stack.block_type_ind != -1
    real_bt_inds64 = pose_stack.block_type_ind[is_real_block].to(torch.int64)
    cf_res_types[is_real_block] = pbt.bt_ind_to_canonical_ind[real_bt_inds64]

    # not necessarily all blocks in the pose are representable
    # in the canonical form; in the future, this will not
    # necessarily be true, that any block type you can put in
    # a PoseStack you can also put in a canonical form, BUT,
    # for right now, canonical form only includes
    # the canonical amino acids
    is_cf_real_res = cf_res_types != -1

    expanded_coords, _ = pose_stack.expand_coords()
    block_atom_is_real = torch.zeros(
        (n_poses, max_n_res, max_n_atoms_per_res), dtype=torch.bool, device=device
    )
    block_atom_is_real[is_real_block] = pbt.atom_is_real[real_bt_inds64]
    (
        nz_pose_ind_for_real_atom,
        nz_res_ind_for_real_atom,
        nz_block_atom_ind_for_real_atom,
    ) = torch.nonzero(block_atom_is_real, as_tuple=True)
    # block_coord_is_real = block_atom_is_real.unsqueeze(3)
    # block_coord_is_real = block_coord_is_real.expand(-1, -1, -1, 3)

    canonical_atom_inds = torch.full(
        (n_poses, max_n_res, max_n_canonical_atoms),
        -1,
        dtype=torch.int32,
        device=device,
    )
    is_real_canonical_atom = torch.zeros(
        (n_poses, max_n_res, max_n_canonical_atoms), dtype=torch.bool, device=device
    )
    canonical_atom_inds_for_bt = pbt.canonical_atom_ind_map[real_bt_inds64]
    is_real_canonical_atom_ind_for_bt = canonical_atom_inds_for_bt != -1
    real_canonical_atom_inds_for_bt = canonical_atom_inds_for_bt[
        is_real_canonical_atom_ind_for_bt
    ]
    # is_real_canonical_atom[
    #     nz_pose_ind_for_real_atom,
    #     nz_block_ind_for_real_atom,
    #     canonical_atom_inds
    # ] = True
    # canonical_atom_inds[is_real_canonical_atom] = pbt.canonical_atom_ind_map[pose_stack.block_type_ind[is_real_block]]
    # is_real_canonical_atom = canonical_atom_inds != -1
    # real_atom_inds = canonical_atom_inds[is_real_canonical_atom]

    cf_coords = torch.full(
        (n_poses, max_n_res, max_n_canonical_atoms, 3),
        torch.nan,
        dtype=torch.float32,
        device=device,
    )
    # print("nz_pose_ind_for_real_atom", nz_pose_ind_for_real_atom.shape)
    # print("nz_res_ind_for_real_atom", nz_res_ind_for_real_atom.shape)
    # print("real_canonical_atom_inds_for_bt", real_canonical_atom_inds_for_bt.shape)
    # print("block_atom_is_real", block_atom_is_real.shape)
    # print("expanded coords", expanded_coords.shape)
    # print("expanded coords[block_atom_is_real]", expanded_coords[block_atom_is_real].shape)

    cf_coords[
        nz_pose_ind_for_real_atom,
        nz_res_ind_for_real_atom,
        real_canonical_atom_inds_for_bt,
    ] = expanded_coords[block_atom_is_real]
    atom_is_present = torch.zeros(
        (n_poses, max_n_res, max_n_canonical_atoms), dtype=torch.bool, device=device
    )
    atom_is_present[
        nz_pose_ind_for_real_atom,
        nz_res_ind_for_real_atom,
        real_canonical_atom_inds_for_bt,
    ] = True

    disulfide_conn_ind = torch.full(
        (n_poses, max_n_res), -1, dtype=torch.int64, device=device
    )
    disulfide_conn_ind[is_real_block] = pbt.canonical_dslf_conn_ind[real_bt_inds64]
    is_real_disulfide_conn = disulfide_conn_ind != -1
    nz_pose_ind_for_dslf, nz_res_ind_for_dslf = torch.nonzero(
        is_real_disulfide_conn, as_tuple=True
    )
    dslf_partner = pose_stack.inter_residue_connections[
        nz_pose_ind_for_dslf,
        nz_res_ind_for_dslf,
        disulfide_conn_ind[is_real_disulfide_conn],
        0,
    ]

    def _u1(x):
        return x.unsqueeze(1)

    redundant_dslf_tuples = torch.cat(
        (_u1(nz_pose_ind_for_dslf), _u1(nz_res_ind_for_dslf), _u1(dslf_partner)), dim=1
    )
    # print("nz_pose_ind_for_dslf", nz_pose_ind_for_dslf.shape)
    # print("redundant_dslf_tuples", redundant_dslf_tuples.shape)
    is_non_redundant_dslf_tuple = (
        redundant_dslf_tuples[:, 1] < redundant_dslf_tuples[:, 2]
    )
    disulfides = redundant_dslf_tuples[is_non_redundant_dslf_tuple, :]

    if chain_id is None:
        chain_id = torch.tensor(
            chain_inds_for_pose_stack(pose_stack), dtype=torch.int32, device=device
        )
    else:
        assert chain_id.shape == (n_poses, max_n_res)

    res_not_connected = determine_res_not_connected_from_pose_stack(
        pose_stack, chain_id, is_real_block, real_bt_inds64
    )

    return (
        chain_id,
        cf_res_types,
        cf_coords,
        atom_is_present,
        disulfides,
        res_not_connected,
    )


def _annotate_packed_block_types_w_termini_types(pbt: PackedBlockTypes):
    # TEMP!
    # for now: a bt is a down/up terminus if it includes
    # "nterm" or "cterm" in its name; these are not good
    # criteria and will be shortly replaced with the more
    # general CanonicalOrdering class that looks at the
    # PatchedChemicalDatabase and the Patches that define
    # the removal of up and down connections
    if hasattr(pbt, "is_term_bt"):
        return
    is_term_bt = torch.zeros((pbt.n_types, 2), dtype=torch.bool)
    for i, bt in enumerate(pbt.active_block_types):
        patches = bt.name.partition(":")
        if "nterm" in patches:
            is_term_bt[i, 0] = True
        if "cterm" in patches:
            is_term_bt[i, 1] = True
    setattr(pbt, "is_term_bt", is_term_bt.to(pbt.device))


def determine_res_not_connected_from_pose_stack(
    pose_stack, chain_id, is_real_block, real_bt_inds64
):
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_types_w_termini_types(pbt)
    # now let's figure out which residues are either a)
    # part of the same chain with i-1 or i+1 but do not
    # have a chemical bound to them or b) are the first/
    # last residue of a chain but are not termini
    n_poses = pose_stack.n_poses
    max_n_res = pose_stack.max_n_blocks
    device = pbt.device

    is_term_block = torch.zeros(
        (n_poses, max_n_res, 2), dtype=torch.bool, device=device
    )
    is_term_block[is_real_block] = pbt.is_term_bt[real_bt_inds64]
    is_chain_first = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    is_chain_last = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    is_chain_first[:, 1:] = chain_id[:, :-1] != chain_id[:, 1:]
    is_chain_first[:, 0] = chain_id[:, 0] != -1
    is_chain_last[:, :-1] = chain_id[:, :-1] != chain_id[:, 1:]
    is_chain_last[:, -1] = chain_id[:, -1] != -1

    is_res_disconnected_from_prev = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    is_res_disconnected_from_next = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    res_has_down_conn = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    res_down_conn = torch.zeros((n_poses, max_n_res), dtype=torch.int64, device=device)
    res_down_conn[is_real_block] = pbt.down_conn_inds[real_bt_inds64].to(torch.int64)
    res_has_down_conn[is_real_block] = res_down_conn[is_real_block] != -1
    res_has_up_conn = torch.zeros(
        (n_poses, max_n_res),
        dtype=torch.bool,
        device=device,
    )
    res_up_conn = torch.zeros((n_poses, max_n_res), dtype=torch.int64, device=device)
    res_up_conn[is_real_block] = pbt.up_conn_inds[real_bt_inds64].to(torch.int64)
    res_has_up_conn[is_real_block] = res_up_conn[is_real_block] != -1

    nz_down_conn_pose_ind, nz_down_conn_res_ind = torch.nonzero(
        res_has_down_conn, as_tuple=True
    )
    nz_up_conn_pose_ind, nz_up_conn_res_ind = torch.nonzero(
        res_has_up_conn, as_tuple=True
    )

    res_arange = (
        torch.arange(max_n_res, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(n_poses, -1)
    )
    res_next_ind = res_arange + 1
    res_prev_ind = res_arange - 1
    # print("nz_down_conn_pose_ind", nz_down_conn_pose_ind.shape)
    # print("nz_down_conn_res_ind", nz_down_conn_res_ind.shape)
    # print("res_down_conn", res_down_conn.shape)
    # print("res_has_down_conn", res_has_down_conn.shape)
    # print("res_down_conn[res_has_down_conn]", res_down_conn[res_has_down_conn].shape)
    # print("pose_stack.inter_residue_connections", pose_stack.inter_residue_connections.shape)
    # print("pose_stack.inter_residue_connections[nz_down_conn_pose_ind,nz_down_conn_res_ind,res_down_conn[res_has_down_conn]]"
    #       , pose_stack.inter_residue_connections[
    #         nz_down_conn_pose_ind,
    #         nz_down_conn_res_ind,
    #         res_down_conn[res_has_down_conn]
    #     ].shape
    #       )
    # print("res_prev_ind", res_prev_ind.shape)
    # print("nz_down_conn_pose_ind", nz_down_conn_pose_ind.shape)
    # print("nz_down_conn_res_ind", nz_down_conn_res_ind.shape)
    # print("res_prev_ind[nz_down_conn_pose_ind,nz_down_conn_res_ind]", res_prev_ind[nz_down_conn_pose_ind,nz_down_conn_res_ind].shape)
    is_res_disconnected_from_prev[nz_down_conn_pose_ind, nz_down_conn_res_ind] = (
        pose_stack.inter_residue_connections[
            nz_down_conn_pose_ind,
            nz_down_conn_res_ind,
            res_down_conn[res_has_down_conn],
            0,
        ]
        != res_prev_ind[nz_down_conn_pose_ind, nz_down_conn_res_ind]
    )
    is_res_disconnected_from_next[nz_up_conn_pose_ind, nz_up_conn_res_ind] = (
        pose_stack.inter_residue_connections[
            nz_up_conn_pose_ind, nz_up_conn_res_ind, res_up_conn[res_has_up_conn], 0
        ]
        != res_next_ind[nz_up_conn_pose_ind, nz_up_conn_res_ind]
    )

    res_not_connected = torch.zeros(
        (n_poses, max_n_res, 2), dtype=torch.bool, device=device
    )
    res_not_connected[:, :, 0] = torch.logical_or(
        torch.logical_and(
            torch.logical_not(is_chain_first), is_res_disconnected_from_prev
        ),
        torch.logical_and(is_chain_first, res_has_down_conn),
    )
    res_not_connected[:, :, 1] = torch.logical_or(
        torch.logical_and(
            torch.logical_not(is_chain_last), is_res_disconnected_from_next
        ),
        torch.logical_and(is_chain_last, res_has_up_conn),
    )

    return res_not_connected
