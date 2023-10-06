import torch

from tmol.pose.pose_stack import PoseStack
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms_v2,
    max_n_canonical_atoms,
)
from tmol.io.details.select_from_canonical import (
    _annotate_packed_block_types_w_canonical_res_order,
    _annotate_packed_block_types_w_dslf_conn_inds,
    _annotate_packed_block_types_w_canonical_atom_order,
)


def canonical_form_from_pose_stack(pose_stack: PoseStack):
    pbt = pose_stack.packed_block_types
    device = pose_stack.device
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    _annotate_packed_block_types_w_canonical_atom_order(pbt)

    n_poses = pose_stack.n_poses
    max_n_res = pose_stack.max_n_blocks
    max_n_atoms_per_res = pbt.max_n_atoms

    cf_coords = torch.zeros(
        (n_poses, max_n_res, max_n_atoms_per_res, 3), dtype=torch.float32, device=device
    )
    cf_res_types = torch.full(
        (n_poses, max_n_res), -1, dtype=torch.int32, device=device
    )

    # not all blocks in the pose are real
    is_real_block = pose_stack.block_type_ind != -1
    cf_res_types[is_real_block] = pbt.bt_ind_to_canonical_ind[
        pose_stack.blcok_type_ind[is_real_block]
    ]
    # not necessarily all blocks in the pose are representable
    # in the canonical form; in the future, this will not
    # necessarily be true, that any block type you can put in
    # a PoseStack you can also put in a canonical form, BUT,
    # for right now, canonical form only includes
    # the canonical amino acids
    is_cf_real_res = cf_res_types != -1

    expanded_coords = pose_stack.expand_coords()
    bt_atom_is_real = torch.zeros(
        (n_poses, max_n_res, max_n_atoms_per_res), dtype=torch.bool, device=device
    )
    real_bt_inds = pose_stack.block_type_ind[is_real_block]
    bt_atom_is_real[is_real_block] = pbt.atom_is_real[real_bt_inds]
    (
        nz_pose_ind_for_real_atom,
        nz_bt_ind_for_real_atom,
        nz_bt_atom_ind_for_real_atom,
    ) = torch.nonzero(bt_atom_is_real, as_tuple=True)

    canonical_atom_inds = torch.full(
        (n_poses, max_n_blocks, max_n_canonical_atoms),
        -1,
        dtype=torch.int32,
        device=device,
    )
    is_real_canonical_atom = torch.zeros(
        (n_poses, max_n_blocks, max_n_canonical_atoms), dtype=torch.bool, device=device
    )
    canonical_atom_inds_for_bt = pbt.canonical_atom_ind_map[real_bt_inds]
    is_real_canonical_atom_ind_for_bt = canonical_atom_inds_for_bt != -1
    real_canonical_atom_inds_for_bt = canonical_atom_inds_for_bt[
        is_real_canonical_atom_ind_for_bt
    ]
    # is_real_canonical_atom[
    #     nz_pose_ind_for_real_atom,
    #     nz_bt_ind_for_real_atom,
    #     canonical_atom_inds
    # ] = True
    # canonical_atom_inds[is_real_canonical_atom] = pbt.canonical_atom_ind_map[pose_stack.block_type_ind[is_real_block]]
    # is_real_canonical_atom = canonical_atom_inds != -1

    real_atom_inds = canonical_atom_inds[is_real_canonical_atom]
    cf_coords[
        nz_pose_ind_for_real_atoms,
        nz_res_ind_for_real_atoms,
        real_canonical_atom_inds_for_bt,
    ] = expanded_coords[bt_atom_is_real]

    disulfide_conn_ind = torch.full(
        (n_poses, max_n_res), -1, dtype=torch.int64, device=device
    )
    disulfide_conn_ind[is_real_block] = pbt.canonical_dslf[real_bt_inds]

    nz_pose_ind_for_dslf, nz_res_ind_for_dslf = torch.nonzero(
        disulfide_conn_ind != -1, as_tuple=True
    )

    return (
        chain_id,
        cf_res_types,
        cf_coords,
    )
