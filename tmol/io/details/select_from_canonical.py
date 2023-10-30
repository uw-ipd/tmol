import numpy
import torch
import attr

from typing import Optional, Tuple
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import (
    CanonicalOrdering,
)

# from tmol.io.details.disulfide_search import cys_co_aa_ind
from tmol.io.details.his_taut_resolution import (
    #     his_co_aa_ind,
    his_taut_variant_ND1_protonated,
)
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder


@validate_args
def assign_block_types(
    canonical_ordering: CanonicalOrdering,
    packed_block_types: PackedBlockTypes,
    atom_is_present: Tensor[torch.bool][:, :, :],
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
    found_disulfides64: Tensor[torch.int64][:, 3],
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
) -> Tuple[
    Tensor[torch.int64][:, :],
    Tensor[torch.int64][:, :, :, 2],
    Tensor[torch.int32][:, :, :, :, :],
]:
    # TEMP!
    torch.set_printoptions(threshold=10000)

    # print("res_types")
    # print(res_types)

    pbt = packed_block_types
    _annotate_packed_block_types_w_canonical_res_order(canonical_ordering, pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    PoseStackBuilder._annotate_pbt_w_polymeric_down_up_bondsep_dist(pbt)
    PoseStackBuilder._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)

    device = pbt.device
    n_poses = chain_id.shape[0]
    max_n_res = chain_id.shape[1]
    max_n_conn = pbt.max_n_conn

    can_ann = pbt.canonical_ordering_annotation
    res_types64 = res_types.to(torch.int64)
    res_type_variants64 = res_type_variants.to(torch.int64)
    is_real_res = res_types64 != -1
    nz_is_real_res = torch.nonzero(is_real_res)

    if res_not_connected is None:
        res_not_connected = torch.zeros(
            (n_poses, max_n_res, 2), dtype=torch.bool, device=device
        )

    (
        termini_variants,
        is_chain_first_res,
        is_chain_last_res,
    ) = determine_chain_ending_status(pbt, chain_id, res_not_connected, is_real_res)

    block_type_ind64 = select_best_block_type_candidate(
        canonical_ordering,
        pbt,
        atom_is_present,
        is_real_res,
        nz_is_real_res,
        res_types64,
        termini_variants,
        res_type_variants64,
    )

    # UGH: stealing/duplicating a lot of code from pose_stack_builder below
    # SHOULD THIS JUST GO IN POSE_STACK_BUILDER AND REPLACE ITS EXISTING CODE???
    # SHOULD POSE_STACK_BUILDER BE DEPRECATED??
    inter_residue_connections64 = torch.full(
        (n_poses, max_n_res, max_n_conn, 2), -1, dtype=torch.int64, device=device
    )

    # is a residue both real and connected to the previous residue?
    res_is_real_and_conn_to_prev = torch.logical_and(
        is_real_res,
        torch.logical_and(
            torch.logical_not(is_chain_first_res),
            torch.logical_not(res_not_connected[:, :, 0]),
        ),
    )
    res_is_real_and_conn_to_next = torch.logical_and(
        is_real_res,
        torch.logical_and(
            torch.logical_not(is_chain_last_res),
            torch.logical_not(res_not_connected[:, :, 1]),
        ),
    )

    connected_up_conn_inds = pbt.up_conn_inds[
        block_type_ind64[res_is_real_and_conn_to_next]
    ].to(torch.int64)
    connected_down_conn_inds = pbt.down_conn_inds[
        block_type_ind64[res_is_real_and_conn_to_prev]
    ].to(torch.int64)

    (
        nz_res_is_real_and_conn_to_prev_pose_ind,
        nz_res_is_real_and_conn_to_prev_res_ind,
    ) = torch.nonzero(res_is_real_and_conn_to_prev, as_tuple=True)
    (
        nz_res_is_real_and_conn_to_next_pose_ind,
        nz_res_is_real_and_conn_to_next_res_ind,
    ) = torch.nonzero(res_is_real_and_conn_to_next, as_tuple=True)

    # now let's mark for each upper-connect the residue and
    # connection id it's connected to
    inter_residue_connections64[
        nz_res_is_real_and_conn_to_next_pose_ind,
        nz_res_is_real_and_conn_to_next_res_ind,
        connected_up_conn_inds,
        0,  # residue id
    ] = nz_res_is_real_and_conn_to_prev_res_ind
    inter_residue_connections64[
        nz_res_is_real_and_conn_to_next_pose_ind,
        nz_res_is_real_and_conn_to_next_res_ind,
        connected_up_conn_inds,
        1,  # connection id
    ] = connected_down_conn_inds

    # now let's mark for each lower-connect the residue and
    # connection id it's connected to
    inter_residue_connections64[
        nz_res_is_real_and_conn_to_prev_pose_ind,
        nz_res_is_real_and_conn_to_prev_res_ind,
        connected_down_conn_inds,
        0,  # residue id
    ] = nz_res_is_real_and_conn_to_next_res_ind
    inter_residue_connections64[
        nz_res_is_real_and_conn_to_prev_pose_ind,
        nz_res_is_real_and_conn_to_prev_res_ind,
        connected_down_conn_inds,
        1,  # connection id
    ] = connected_up_conn_inds

    # if we have any disulfides, then we need to also mark those
    # connections in the inter_residue_connections64 map
    if found_disulfides64.shape[0] != 0:
        cyd1_block_type64 = block_type_ind64[
            found_disulfides64[:, 0], found_disulfides64[:, 1]
        ]
        cyd2_block_type64 = block_type_ind64[
            found_disulfides64[:, 0], found_disulfides64[:, 2]
        ]

        # n- and c-term CYD residues will have different dslf connection inds
        # than mid-cyd residues; therefore we don't just hard code "2" here
        cyd1_dslf_conn64 = pbt.canonical_dslf_conn_ind[cyd1_block_type64].to(
            torch.int64
        )
        cyd2_dslf_conn64 = pbt.canonical_dslf_conn_ind[cyd2_block_type64].to(
            torch.int64
        )

        inter_residue_connections64[
            found_disulfides64[:, 0], found_disulfides64[:, 1], cyd1_dslf_conn64, 0
        ] = found_disulfides64[:, 2]
        inter_residue_connections64[
            found_disulfides64[:, 0], found_disulfides64[:, 1], cyd1_dslf_conn64, 1
        ] = cyd2_dslf_conn64
        inter_residue_connections64[
            found_disulfides64[:, 0], found_disulfides64[:, 2], cyd2_dslf_conn64, 0
        ] = found_disulfides64[:, 1]
        inter_residue_connections64[
            found_disulfides64[:, 0], found_disulfides64[:, 2], cyd2_dslf_conn64, 1
        ] = cyd1_dslf_conn64

    # now that we have the inter-residue connections established,
    # proceed with the rest of the PoseStackBuilder's steps
    # in constructing the inter_block_bondsep tensor using the
    # all-pairs-shortest-path algorithm
    # 3a
    (
        pconn_matrix,
        pconn_offsets,
        block_n_conn,
        pose_n_pconn,
    ) = PoseStackBuilder._take_real_conn_conn_intrablock_pairs(
        pbt, block_type_ind64, is_real_res
    )

    # 3b
    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        inter_residue_connections64, pconn_offsets, pconn_matrix
    )

    # # SHORT CIRCUIT: skip the all-pairs-shortest-path call
    # inter_block_bondsep64 = torch.full(
    #     (n_poses, max_n_res, max_n_res, max_n_conn, max_n_conn),
    #     6,
    #     dtype=torch.int64,
    #     device=pbt.device,
    # )

    # 4
    # bad naming because python is annoying that way:
    # inter_block_bondsep64 <= ibb64
    ibb64 = PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph(
        pbt, block_n_conn, pose_n_pconn, pconn_matrix
    )

    # print("end assign_block_types")
    return (block_type_ind64, inter_residue_connections64, ibb64)


@validate_args
def determine_chain_ending_status(
    pbt: PackedBlockTypes,
    chain_id: Tensor[torch.int32][:, :],
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]],
    is_real_res: Tensor[torch.bool][:, :],
):
    n_poses = chain_id.shape[0]
    max_n_res = chain_id.shape[1]
    max_n_conn = pbt.max_n_conn
    device = pbt.device

    # logic for deciding what chemical bonds are present between the polymeric
    # residues and which residues should be represented as termini:
    # - The first and last residues in a chain are not connected to the
    #   previous / next residues
    # - If res_not_connected[p, i, 0] is true, then this residue will not
    #   be connected to i-1;
    # - If res_not_connected[p, i, 1] is true, then this residue will not
    #   be connected to i+1
    # - Thus: for residue i and i+1 to be connected, all the following must hold:
    #   - i cannot be the last residue in a chain
    #   - (equivalently, i+1 cannot be the first residue in a chain)
    #   - res_not_connected[p, i, 1] must be false
    #   - res_not_connected[p, i+1, 0] must be false
    # - If a residue is the first residue in a chain and res_not_connected[p, i, 0]
    #   is true, then it will not be treated as an n-terminal residue, rather,
    #   it will be treated as a "mid" residue (unless it is also a c-term residue)
    #   and, in the case of amino acids, will have a regular H atom bonded to N
    #   instead of the 1H, 2H, and 3H atoms. Its connection to the "previous" residue
    #   will be incomplete (-1) instead of to a particular residue.
    # - If a residue is the last residue in a chain and res_not_connected[p, i, 1]
    #   is true, then it will not be treated as a c-terminal residue.
    is_chain_first_res = torch.logical_and(
        is_real_res,
        torch.cat(
            (
                torch.ones((n_poses, 1), dtype=torch.bool, device=device),
                chain_id[:, 1:] != chain_id[:, :-1],  # is i different from i-1?
            ),
            dim=1,
        ),
    )
    is_chain_last_res = torch.logical_and(
        is_real_res,
        torch.cat(
            (
                chain_id[:, :-1] != chain_id[:, 1:],  # is i different from i+1?
                torch.ones((n_poses, 1), dtype=torch.bool, device=device),
            ),
            dim=1,
        ),
    )
    is_down_term_res = torch.logical_and(
        is_chain_first_res, torch.logical_not(res_not_connected[:, :, 0])
    )
    is_up_term_res = torch.logical_and(
        is_chain_last_res, torch.logical_not(res_not_connected[:, :, 1])
    )
    is_down_and_not_up_term_res = torch.logical_and(
        is_down_term_res, torch.logical_not(is_up_term_res)
    )
    is_up_and_not_down_term_res = torch.logical_and(
        is_up_term_res, torch.logical_not(is_down_term_res)
    )
    is_both_down_and_up_term_res = torch.logical_and(is_down_term_res, is_up_term_res)

    # note which polymeric residues should be given their chain-ending
    # (aka termini) patches
    termini_variants = torch.ones_like(chain_id, dtype=torch.int64)
    termini_variants[is_down_and_not_up_term_res] = 0
    termini_variants[is_up_and_not_down_term_res] = 2
    termini_variants[is_both_down_and_up_term_res] = 3

    return termini_variants, is_chain_first_res, is_chain_last_res


@validate_args
def select_best_block_type_candidate(
    canonical_ordering: CanonicalOrdering,
    pbt: PackedBlockTypes,
    atom_is_present: Tensor[torch.bool][:, :, :],
    is_real_res: Tensor[torch.bool][:, :],
    nz_is_real_res: Tensor[torch.int64][:, :],
    res_types64: Tensor[torch.int64][:, :],
    termini_variants: Tensor[torch.int64][:, :],
    res_type_variants64: Tensor[torch.int64][:, :],
):
    device = pbt.device
    n_poses = atom_is_present.shape[0]
    max_n_res = atom_is_present.shape[1]
    can_ann = pbt.canonical_ordering_annotation
    max_n_candidates = can_ann.max_n_candidates_for_var_combo

    block_type_candidates = torch.full(
        (n_poses, max_n_res, can_ann.max_n_candidates_for_var_combo),
        -1,
        dtype=torch.int64,
        device=device,
    )
    is_real_candidate = torch.zeros(
        (n_poses, max_n_res, can_ann.max_n_candidates_for_var_combo),
        dtype=torch.bool,
        device=device,
    )
    block_type_candidates[is_real_res] = can_ann.var_combo_candidate_bt_index[
        res_types64[is_real_res],
        termini_variants[is_real_res],
        res_type_variants64[is_real_res],
    ]

    is_real_candidate[is_real_res] = can_ann.var_combo_is_real_candidate[
        res_types64[is_real_res],
        termini_variants[is_real_res],
        res_type_variants64[is_real_res],
    ]
    provided_atoms_absent_from_candidate = torch.zeros(
        (
            n_poses,
            max_n_res,
            can_ann.max_n_candidates_for_var_combo,
            canonical_ordering.max_n_canonical_atoms,
        ),
        dtype=torch.bool,
        device=device,
    )
    atom_is_present = atom_is_present.unsqueeze(2).expand(
        -1, -1, can_ann.max_n_candidates_for_var_combo, -1
    )
    candidate_atom_is_absent = torch.zeros(
        (
            n_poses,
            max_n_res,
            can_ann.max_n_candidates_for_var_combo,
            canonical_ordering.max_n_canonical_atoms,
        ),
        dtype=torch.bool,
        device=device,
    )
    candidate_atom_is_absent[is_real_candidate] = can_ann.bt_canonical_atom_is_absent[
        block_type_candidates[is_real_candidate]
    ]
    # print("var_atom_is_absent")
    # print(var_atom_is_absent)
    provided_atoms_absent_from_candidate[is_real_candidate] = torch.logical_and(
        atom_is_present[is_real_candidate],
        candidate_atom_is_absent[is_real_candidate],
    )

    # if there are any atoms that were provided for a given residue
    # but that the variant does not contain, then that is not a match
    exclude_candidate_for_residue = torch.any(
        provided_atoms_absent_from_candidate, dim=3
    )
    atom_is_absent = torch.logical_not(atom_is_present)
    candidate_non_term_patch_atom_is_present = torch.zeros_like(
        candidate_atom_is_absent
    )
    candidate_non_term_patch_atom_is_present[
        is_real_candidate
    ] = can_ann.bt_non_term_patch_added_canonical_atom_is_present[
        block_type_candidates[is_real_candidate]
    ]

    canonical_atom_was_not_provided_for_candidate = torch.zeros_like(
        provided_atoms_absent_from_candidate, dtype=torch.int64
    )
    canonical_atom_was_not_provided_for_candidate[
        is_real_candidate
    ] = torch.logical_and(
        atom_is_absent[is_real_candidate],
        candidate_non_term_patch_atom_is_present[is_real_candidate],
    ).to(
        torch.int64
    )
    candidate_is_non_default_term = torch.zeros(
        (n_poses, max_n_res, can_ann.max_n_candidates_for_var_combo),
        dtype=torch.int64,
        device=device,
    )
    candidate_is_non_default_term[
        is_real_candidate
    ] = can_ann.bt_is_non_default_terminus[block_type_candidates[is_real_candidate]].to(
        torch.int64
    )

    # for i in range(n_poses):
    #     for j in range(max_n_res):
    #         for k in range(canonical_atom_was_not_provided_for_variant.shape[2]):
    #             if not is_real_candidate[i, j, k]:
    #                 continue
    #             which_bt = block_type_candidates[i, j, k]
    #             cand_bt = pbt.active_block_types[which_bt]
    #             print(i, j, k, which_bt.item(), cand_bt.name, "restype", res_types[i, j], "equiv class", canonical_ordering.restype_io_equiv_classes[res_types[i,j]])
    #             for l in range(canonical_atom_was_not_provided_for_variant.shape[3]):
    #                 if canonical_atom_was_not_provided_for_variant[i, j, k, l]:
    #                     print(" atom not provided:", cand_bt.atoms[l].name)

    n_canonical_atoms_not_provided_for_candidate = torch.sum(
        canonical_atom_was_not_provided_for_candidate, dim=3
    )
    n_canonical_atoms_not_provided_for_candidate[
        torch.logical_or(
            torch.logical_not(is_real_candidate), exclude_candidate_for_residue
        )
    ] = (canonical_ordering.max_n_canonical_atoms + 1)
    # print("n_canonical_atoms_not_provided_for_var")
    # print(n_canonical_atoms_not_provided_for_var)

    candidate_misalignment_score = (
        2 * n_canonical_atoms_not_provided_for_candidate + candidate_is_non_default_term
    )
    best_candidate_ind = torch.argmin(candidate_misalignment_score, dim=2)

    # ok, we need to do some quality checks. If the best fit variant's score is
    # 2 * (canonical_ordering.max_n_canonical_atoms + 1) or worse, then we have
    # a problem. It's hard to know what to do at this point!
    best_candidate_score = torch.zeros(
        (n_poses, max_n_res), dtype=torch.int64, device=device
    )
    best_candidate_score[is_real_res] = candidate_misalignment_score[
        nz_is_real_res[:, 0],
        nz_is_real_res[:, 1],
        best_candidate_ind[is_real_res],
    ]

    failure_score = 2 * (canonical_ordering.max_n_canonical_atoms + 1)
    # print("failures")
    # print(torch.nonzero(best_candidate_score >= failure_score))
    # print("any?")
    # print(torch.any(best_candidate_score >= failure_score))

    if torch.any(best_candidate_score >= failure_score):
        for i in range(n_poses):
            for j in range(max_n_res):
                if best_candidate_score[i, j] < failure_score:
                    continue
                print("best candidate for failure:", best_candidate_ind[i, j].item())
                for k in range(max_n_candidates):
                    print("candidate", k, " real? ", is_real_candidate[i, j, k].item())
                    if not is_real_candidate[i, j, k]:
                        continue
                    which_bt = block_type_candidates[i, j, k]
                    cand_bt = pbt.active_block_types[which_bt]
                    print(
                        i,
                        j,
                        k,
                        which_bt.item(),
                        cand_bt.name,
                        "restype",
                        res_types[i, j],
                        "equiv class",
                        canonical_ordering.restype_io_equiv_classes[res_types[i, j]],
                    )
                    if exclude_candidate_for_residue[i, j, k]:
                        equiv_class = bt.io_equiv_class
                        for l in range(canonical_ordering.max_n_canonical_atoms):
                            if provided_atoms_absent_from_candidate[i, j, k, l]:
                                print(
                                    " atom",
                                    canonical_ordering.restypes_ordered_atom_names[
                                        equiv_class
                                    ][l],
                                    "provided but absent from candidate",
                                    bt.name,
                                )
                    else:
                        for l in range(canonical_ordering.max_n_canonical_atoms):
                            if canonical_atom_was_not_provided_for_candidate[
                                i, j, k, l
                            ]:
                                print(" atom not provided:", cand_bt.atoms[l].name)

        print("Failed to find a matching block type")
        # print(best_fit_variant_score)
        # print("bad")
        # print(torch.nonzero(best_fit_variant_score >= 2 * (canonical_ordering.max_n_canonical_atoms + 1)))
        raise RuntimeError(
            "failed to resolve a block type from the candidates available"
        )

    block_type_ind64 = torch.full_like(res_types64, -1)
    block_type_ind64[is_real_res] = block_type_candidates[
        nz_is_real_res[:, 0],
        nz_is_real_res[:, 1],
        best_candidate_ind[is_real_res],
    ]

    return block_type_ind64


@validate_args
def take_block_type_atoms_from_canonical(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.bool][:, :, :],
):
    """Now that we have decided which block type each canonical residue
    is, we want to select only those atoms from the canonically-ordered
    coords and atom_is_present tensors
    In the case of the atom_is_present tensor, we will here forward only
    concern ourselves with the atoms that are missing and not with
    the (perhaps more than one) ways in which an atom can be (tentatively)
    provided; thus we will invert atom_is_present and return the new
    tensor as missing_atoms
    """
    pbt = packed_block_types
    assert hasattr(pbt, "canonical_ordering_annotation")
    can_ann = pbt.canonical_ordering_annotation
    device = pbt.device
    # _annotate_packed_block_types_w_canonical_res_order(pbt)
    # _annotate_packed_block_types_w_canonical_atom_order(pbt)

    n_poses = block_types64.shape[0]
    max_n_blocks = block_types64.shape[1]
    real_block_types = block_types64 != -1
    real_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=pbt.device
    )
    real_atoms[real_block_types] = pbt.atom_is_real[block_types64[real_block_types]]

    canonical_atom_inds = torch.full(
        (n_poses, max_n_blocks, pbt.max_n_atoms),
        -1,
        dtype=torch.int64,
        device=pbt.device,
    )

    canonical_atom_inds[real_block_types] = can_ann.bt_canonical_atom_ind_map[
        block_types64[real_block_types]
    ]
    nz_real_pose_ind, nz_real_block_ind, _ = torch.nonzero(real_atoms, as_tuple=True)

    block_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3), dtype=torch.float32, device=device
    )
    missing_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=pbt.device
    )
    real_canonical_atom_inds = canonical_atom_inds[real_atoms]
    block_coords[real_atoms] = coords[
        nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds
    ]
    missing_atoms[real_atoms] = torch.logical_not(
        atom_is_present[nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds]
    )

    return (block_coords, missing_atoms, real_atoms, real_canonical_atom_inds)


@attr.s(auto_attribs=True, frozen=True)
class CanonicalOrderingAnnotation:
    max_n_candidates_for_var_combo: int
    # n-co-equiv-class x n-term-opts x n-spcase-var
    var_combo_n_candidates: Tensor[torch.int64][:, :, :]
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_is_real_candidate: Tensor[torch.bool][:, :, :, :]
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_candidate_bt_index: Tensor[torch.int64][:, :, :, :]
    # n-pbt-block-types x max-n-canonical-atoms
    bt_canonical_atom_is_absent: Tensor[torch.bool][:, :]
    # n-pbt-block-types x max-n-canonical-atoms
    bt_non_term_patch_added_canonical_atom_is_present: Tensor[torch.bool][:, :]
    # n-pbt-block-types
    bt_is_non_default_terminus: Tensor[torch.bool][:]

    # n-pbt-block-types
    # needed for output?
    # bt_ind_to_canonical_io_equiv_class_ind: Tensor[torch.int64][:]

    # n-pbt-block-types x max-n-atoms
    bt_canonical_atom_ind_map: Tensor[torch.int64][:, :]


@validate_args
def _annotate_packed_block_types_w_canonical_res_order(
    canonical_ordering, pbt: PackedBlockTypes
):
    # mapping from canonical restype index to the
    # packed-block-types index for that restype
    #
    # what we want here is to be able to take four pieces of data:
    #  a. restype by 3-letter code
    #  b. termini assignment
    #  c. disulfide assignment
    #  d. his-d vs his-e assignment
    # and then look up into a tensor exactly which block type we are talking about

    co = canonical_ordering

    if hasattr(pbt, "canonical_ording_annotation"):
        return

    # what is the problem we are trying to solve?
    # we have a number of "io_equiv_class"es that we want to map
    # to particular block types
    # where the user/the chain connectivity
    # can specify things such as:
    # is it down-term, is it up-term, is it neither
    # is it disulfide-bonded, is it a particular kind of his
    # and the user has given us a list of atoms that are
    # present or absent. These atoms will help us decide
    # which variant the user is requesting, e.g.,
    # phospho-serine by providing a P atom.
    # The algorithm for deciding which block type
    # from a set of candidates will be:
    # from the set of bts with the "appropriate" termini,
    # given the list of provided atoms for a given residue,
    # find the bt whose atom list has all of the provided atoms
    # and is missing the fewest atoms that were not provided
    # e.g. if atoms A, B and C were provided and
    # BT #1 has atoms A and B
    # BT #2 has atoms A B C and D, and
    # BT #3 has atoms A B C D and E, and
    # then the best match is not BT #1 because it does not have
    # provided atom C,
    # and BT #2 is preferred to BT #3 because BT #3 is missing
    # more atoms.
    # so if we have array
    # p  [1, 1, 1, 0, 0] representing provided atoms A, B, and C, and
    # b1 [1, 1, 0, 0, 0] for BT #1, and
    # b2 [1, 1, 1, 1, 0] for BT #2, and
    # b3 [1, 1, 1, 1, 1] for BT #3,
    # then
    # sum((p - b1) == 1) = sum(p & ~b1) ==> 1
    # sum((p - b2) == 1) = sum(p & ~b2) ==> 0
    # sum((p - b3) == 1) = sum(p & ~b3) ==> 0
    # so we would eliminate b1
    # and then
    # sum((b1 - p) == 1) = sum(b1 & ~p) ==> 0  but note this option will have been eliminated
    # sum((b2 - p) == 1) = sum(b2 & ~p) ==> 1
    # sum((b3 - p) == 1) = sum(b3 & ~p) ==> 2
    # so if we take the minimum among the non-eliminated set of b2 and b3
    # that would tell us to assign b2 to this residue.

    # ok, so then, we need to know for each
    # combination of (name3, terminus-assignment, special-case variant)
    # the set of compatible variants
    # and for each variant the set of "canonical atoms" that it contains
    # (e.g. b2) and the set of canonical atoms it does not contain (e.g. ~b2)
    # (though this second set could be computed only as needed.)

    # IDEA 1
    # For each set of block types that all have the same non-termini patches,
    # we want to encode which are patched with the default termini, so that
    # all-else being equal, we can break minimum-missing-atoms ties by
    # assigning the bts patched with the default termini

    # IDEA 2
    # ignore the atoms added by termini patches when counting how many atoms
    # in the "present" set are missing from the bt's set
    # so e.g. if
    # bt1 has atoms [A B C Q R] after its term patch added atoms Q and R, and
    # bt2 has atoms [A B C S ] after its term patch added atom S, and
    # the present set has atoms [A B C], and
    # bt1 has been declared the "default"
    # then both bt1 and bt2 would have the same score of 0 and
    # the tie would go to bt1.
    # logically, this would happen with
    # sum(p & ~b1) as before for looking to make sure all atoms in p are contained in b1
    # but the second part would become
    # sum(b1_sans_termini_additions & ~p) counting only non-termini-patch-added atoms
    # of b1 that are absent from p against b1.
    #
    # how is that going to be encoded???
    # bt1 can be given the "tie breaker" status so that it appears
    # better than bts with the same score, but not better than
    # bt2 with a better score -- i.e. each tie-breaker residue has
    # .25 subtracted from its score, so 0-> -0.25, 1-->0.75,
    # and then we use "min"

    max_n_termini_types = 4  # 0=down-term, 1=mid, 2=up-term, 3=down+up
    max_n_special_case_aa_variant_types = (
        2  # CYS=0, CYD=1; HISE=0, HISD=1; all others, 0
    )

    def map_term_to_int(is_down_term, is_up_term):
        if is_down_term and is_up_term:
            return 3
        if is_down_term:
            return 0
        if is_up_term:
            return 2
        return 1

    def map_spcase_var_to_int(is_cyd, is_hisd):
        # spcase == SPecial CASE
        if is_cyd or is_hisd:
            return 1
        return 0

    def term_and_spcase_var_candidate_lists():
        candidates = []
        for i in range(max_n_termini_types):
            candidates.append([])
            for j in range(max_n_special_case_aa_variant_types):
                candidates[i].append([])
        return candidates

    pbt_io_equiv_class_name_set = set(
        [bt.io_equiv_class for bt in pbt.active_block_types]
    )
    pbt_io_equiv_class_candidates = {
        io_equiv_class: term_and_spcase_var_candidate_lists()
        for io_equiv_class in pbt_io_equiv_class_name_set
    }
    bt_is_non_default_terminus = torch.zeros((pbt.n_types,), dtype=torch.bool)

    # the number of base types in this PBT, which may represent a subset
    # of the base types in the ChemicalDatabase from which the CO was
    # derived
    n_co_io_equiv_classes = co.n_restype_io_equiv_classes
    n_pbt_io_equiv_classes = len(pbt_io_equiv_class_name_set)

    for i, bt in enumerate(pbt.active_block_types):
        bt_vars = bt.name.split(":")
        bt_is_down_term = False
        bt_is_up_term = False
        bt_is_non_default_term = False
        bt_is_cyd = bt.base_name == "CYD"
        bt_is_hisd = bt.base_name == "HIS_D"
        for var_name in bt_vars[1:]:
            if var_name in co.down_termini_patches:
                bt_is_down_term = True
                if (
                    var_name
                    != co.restypes_default_termini_mapping[bt.io_equiv_class][0]
                ):
                    bt_is_non_default_term = True
            if var_name in co.up_termini_patches:
                bt_is_up_term = True
                if (
                    var_name
                    != co.restypes_default_termini_mapping[bt.io_equiv_class][1]
                ):
                    bt_is_non_default_term = True
        term_ind = map_term_to_int(bt_is_down_term, bt_is_up_term)
        spcase_var_ind = map_spcase_var_to_int(bt_is_cyd, bt_is_hisd)
        # if bt.io_equiv_class == "CYS":
        #     print(bt.name, term_ind, spcase_var_ind, i)
        pbt_io_equiv_class_candidates[bt.io_equiv_class][term_ind][
            spcase_var_ind
        ].append((bt, i))
        bt_is_non_default_terminus[i] = bt_is_non_default_term

    max_n_candidates_for_var_combo = max(
        len(pbt_io_equiv_class_candidates[bt][i][j][0])
        for bt in pbt_io_equiv_class_candidates
        for i in range(max_n_termini_types)
        for j in range(max_n_special_case_aa_variant_types)
        if len(pbt_io_equiv_class_candidates[bt][i][j]) > 0
    )

    var_combo_candidate_bt_index = torch.full(
        (
            n_co_io_equiv_classes,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
            max_n_candidates_for_var_combo,
        ),
        -1,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )
    var_combo_is_real_candidate = torch.zeros(
        (
            n_co_io_equiv_classes,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
            max_n_candidates_for_var_combo,
        ),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )
    var_combo_n_candidates = torch.zeros(
        (
            n_co_io_equiv_classes,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
        ),
        dtype=torch.int64,
        device=torch.device("cpu"),
    )
    for i, bt_name3 in enumerate(co.restype_io_equiv_classes):
        if bt_name3 not in pbt_io_equiv_class_candidates:
            continue
        for j in range(max_n_termini_types):
            for k in range(max_n_special_case_aa_variant_types):
                var_combo_n_candidates[i, j, k] = len(
                    pbt_io_equiv_class_candidates[bt_name3][j][k]
                )
                for l, (bt, bt_ind) in enumerate(
                    pbt_io_equiv_class_candidates[bt_name3][j][k]
                ):
                    # if bt_name3 == "CYS":
                    #     print(bt_name3, i, j, k, l, "bt_ind", bt_ind)
                    var_combo_candidate_bt_index[i, j, k, l] = bt_ind
                    var_combo_is_real_candidate[i, j, k, l] = True

    # For bt i and canonical atom j, is canonical atom j absent from bt i?
    # needed so we can compute p & ~b[i]
    bt_canonical_atom_is_absent = torch.ones(
        (
            pbt.n_types,
            co.max_n_canonical_atoms,
        ),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )
    # For bt i and  canonical atom j, is canonical atom j present in bt i
    # and not put there by a termini variant?
    bt_non_term_patch_added_canonical_atom_is_present = torch.zeros(
        (
            pbt.n_types,
            co.max_n_canonical_atoms,
        ),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )
    for i, bt in enumerate(pbt.active_block_types):
        bt_at_names = set([at.name for at in bt.atoms])
        # first mark all atoms that are present as not absent
        # including the termini atoms
        for at_name in bt_at_names:
            can_ind = co.restypes_atom_index_mapping[bt.name3][at_name]
            bt_canonical_atom_is_absent[i, can_ind] = False

        # now we will go and strike out all the atoms that were not
        # added by termini patches so we can mark them as present
        variants = bt.name.split(":")[1:]
        for var in variants:
            for can_at in co.termini_patch_added_atoms[var]:
                if can_at in bt_at_names:
                    bt_at_names.remove(can_at)

        for at_name in bt_at_names:
            can_ind = co.restypes_atom_index_mapping[bt.io_equiv_class][at_name]
            bt_non_term_patch_added_canonical_atom_is_present[i, can_ind] = True
            bt_canonical_atom_is_absent[i, can_ind] = False

    # bt_ind_to_canonical_ind = numpy.array(
    #     [
    #         co.restype_io_equiv_classes.index(bt.io_equiv_class)
    #         for bt in pbt.active_block_types
    #     ],
    #     dtype=numpy.int32
    # )
    bt_canonical_atom_ind = numpy.full(
        (pbt.n_types, pbt.max_n_atoms), -1, dtype=numpy.int64
    )
    for i, bt in enumerate(pbt.active_block_types):
        assert bt.io_equiv_class in co.restypes_ordered_atom_names
        i_canonical_ordering = co.restypes_ordered_atom_names[bt.io_equiv_class]
        for j, at in enumerate(bt.atoms):
            # probably this would be faster if we used a pandas indexer
            # but this is done only once, so, for now, use the slow form
            bt_canonical_atom_ind[i, j] = i_canonical_ordering.index(at.name.strip())
    bt_canonical_atom_ind = torch.tensor(bt_canonical_atom_ind, dtype=torch.int64)

    def _d(x):
        return x.to(pbt.device)

    ann = CanonicalOrderingAnnotation(
        max_n_candidates_for_var_combo=max_n_candidates_for_var_combo,
        var_combo_n_candidates=_d(var_combo_n_candidates),
        var_combo_is_real_candidate=_d(var_combo_is_real_candidate),
        var_combo_candidate_bt_index=_d(var_combo_candidate_bt_index),
        bt_canonical_atom_is_absent=_d(bt_canonical_atom_is_absent),
        bt_non_term_patch_added_canonical_atom_is_present=_d(
            bt_non_term_patch_added_canonical_atom_is_present
        ),
        bt_is_non_default_terminus=_d(bt_is_non_default_terminus),
        # bt_ind_to_canonical_ind=_d(bt_ind_to_canonical_ind),
        bt_canonical_atom_ind_map=_d(bt_canonical_atom_ind),
    )
    setattr(pbt, "canonical_ordering_annotation", ann)

    # variants_for_blocktypes = defaultdict(list)
    # non_default_termini_types = defaultdict(lambda: set([]))
    # for bt in bt.active_block_types:
    #     variants_for_blocktypes[bt.base_name].append(bt)

    # forward ordering: from canonical-index + termini type + variant --> block-type index
    # canonical_ordering_map = numpy.full(
    #     (len(ordered_canonical_aa_types), max_n_termini_types, max_n_aa_variant_types),
    #     -1,
    #     dtype=numpy.int32,
    # )
    # bt_ind_to_canonical_ind = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)
    #
    # def bt_inds_for_variant(base_names, var):
    #     return pbt.restype_index.get_indexer(
    #         [
    #             bt_name if var == "" else ":".join((bt_name, var))
    #             for bt_name in base_names
    #         ]
    #     )

    # canonical_ordering_map[:, 0, 0] = bt_inds_for_variant(
    #     ordered_canonical_aa_types, "nterm"
    # )
    # canonical_ordering_map[:, 1, 0] = bt_inds_for_variant(
    #     ordered_canonical_aa_types, ""
    # )
    # canonical_ordering_map[:, 2, 0] = bt_inds_for_variant(
    #     ordered_canonical_aa_types, "cterm"
    # )

    # NOTE: We have two amino acids with (non-termini) "variants" that we are going to handle
    # CYD, the disulfided CYS is variant 1; CYS is variant 0
    # between HIS and HIS_D, one of them is variant 1 and one is variant 0
    # cys_his_var_names = [
    #     "CYD",
    #     "HIS_D" if his_taut_variant_ND1_protonated == 1 else "HIS",
    # ]
    # cys_his = (cys_co_aa_ind, his_co_aa_ind)
    # canonical_ordering_map[cys_his, 0, 1] = bt_inds_for_variant(
    #     cys_his_var_names, "nterm"
    # )
    # canonical_ordering_map[cys_his, 1, 1] = bt_inds_for_variant(cys_his_var_names, "")
    # canonical_ordering_map[cys_his, 2, 1] = bt_inds_for_variant(
    #     cys_his_var_names, "cterm"
    # )

    # # map from the specific block type to the generic canoncial aa index
    # for i in range(len(canonical_ordering_map)):
    #     for j in range(3):
    #         for k in range(2):
    #             if canonical_ordering_map[i, j, k] != -1:
    #                 bt_ind_to_canonical_ind[canonical_ordering_map[i, j, k]] = i

    # def t(x):
    #     return torch.tensor(x, dtype=torch.int64, device=pbt.device)

    # setattr(pbt, "canonical_res_ordering_map", t(canonical_ordering_map))
    # setattr(pbt, "bt_ind_to_canonical_ind", t(bt_ind_to_canonical_ind))


@validate_args
def _annotate_packed_block_types_w_dslf_conn_inds(pbt: PackedBlockTypes):
    if hasattr(pbt, "canonical_dslf_conn_ind"):
        return
    canonical_dslf_conn_ind = numpy.full((pbt.n_types,), -1, dtype=numpy.int64)
    for i, bt in enumerate(pbt.active_block_types):
        if "dslf" in bt.connection_to_cidx:
            canonical_dslf_conn_ind[i] = bt.connection_to_cidx["dslf"]
    canonical_dslf_conn_ind = torch.tensor(
        canonical_dslf_conn_ind, dtype=torch.int64, device=pbt.device
    )
    setattr(pbt, "canonical_dslf_conn_ind", canonical_dslf_conn_ind)


# @validate_args
# def _annotate_packed_block_types_w_canonical_atom_order(pbt: PackedBlockTypes):
#     if hasattr(pbt, "canonical_atom_ind_map"):
#         return
#     setattr(pbt, "canonical_atom_ind_map", canonical_atom_ind)
