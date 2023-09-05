import numpy
import torch

from typing import Optional, Tuple
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms_v2,
)
from tmol.io.details.disulfide_search import cys_co_aa_ind
from tmol.io.details.his_taut_resolution import (
    his_co_aa_ind,
    his_taut_variant_ND1_protonated,
)
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder


@validate_args
def assign_block_types(
    packed_block_types: PackedBlockTypes,
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
    pbt = packed_block_types
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    PoseStackBuilder._annotate_pbt_w_polymeric_down_up_bondsep_dist(pbt)
    PoseStackBuilder._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)

    device = pbt.device
    n_poses = chain_id.shape[0]
    max_n_res = chain_id.shape[1]
    max_n_conn = pbt.max_n_conn

    # canonica_res_ordering_map dimensioned: [20aas x 3 termini types x 2 variant types]
    # 3 termini types? 0-nterm, 1-mid, 2-cterm
    canonical_res_ordering_map = pbt.canonical_res_ordering_map
    res_types64 = res_types.to(torch.int64)
    res_type_variants64 = res_type_variants.to(torch.int64)
    is_real_res = res_types64 != -1

    if res_not_connected is None:
        res_not_connected = torch.zeros(
            (n_poses, max_n_res, 2), dtype=torch.bool, device=device
        )

    # assert torch.all(
    #     is_real_res[chain_begin == 1]
    # ), "every residue marked as the beginning of a chain must be real"

    # we are going to modify the chain_id tensor, perhaps to the detriment of the caller,
    # so clone it.
    # mark the chain id for any un-real residue as -1
    # then the n-term residues will have a different chain id
    # from their i-1 residues
    chain_id = chain_id.clone()
    chain_id[res_types64 == -1] = -1

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
    chain_first_res = torch.logical_and(
        is_real_res,
        torch.cat(
            (
                torch.ones((n_poses, 1), dtype=torch.bool, device=device),
                chain_id[:, 1:] != chain_id[:, :-1],  # is i different from i-1?
            ),
            dim=1,
        ),
    )
    chain_last_res = torch.logical_and(
        is_real_res,
        torch.cat(
            (
                chain_id[:, :-1] != chain_id[:, 1:],  # is i different from i+1?
                torch.ones((n_poses, 1), dtype=torch.bool, device=device),
            ),
            dim=1,
        ),
    )
    n_term_res = torch.logical_and(
        chain_first_res, torch.logical_not(res_not_connected[:, :, 0])
    )
    c_term_res = torch.logical_and(
        chain_last_res, torch.logical_not(res_not_connected[:, :, 1])
    )

    termini_variants = torch.ones_like(res_types, dtype=torch.int64)
    termini_variants[n_term_res] = 0
    termini_variants[c_term_res] = 2

    block_type_ind64 = torch.full_like(res_types64, -1)
    block_type_ind64[is_real_res] = canonical_res_ordering_map[
        res_types64[is_real_res],
        termini_variants[is_real_res],
        res_type_variants64[is_real_res],
    ]

    # UGH: stealing/duplicating a lot of code from pose_stack_builder below
    inter_residue_connections64 = torch.full(
        (n_poses, max_n_res, max_n_conn, 2), -1, dtype=torch.int64, device=device
    )

    # is a residue both real and connected to the previous residue?
    res_is_real_and_conn_to_prev = torch.logical_and(
        is_real_res,
        torch.logical_and(
            torch.logical_not(chain_first_res),
            torch.logical_not(res_not_connected[:, :, 0]),
        ),
    )
    res_is_real_and_conn_to_next = torch.logical_and(
        is_real_res,
        torch.logical_and(
            torch.logical_not(chain_last_res),
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

    return (block_type_ind64, inter_residue_connections64, ibb64)


@validate_args
def take_block_type_atoms_from_canonical(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.int32][:, :, :],
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
    device = pbt.device
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_canonical_atom_order(pbt)

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

    canonical_atom_inds[real_block_types] = pbt.canonical_atom_ind_map[
        block_types64[real_block_types]
    ]
    nz_real_pose_ind, nz_real_block_ind, _ = torch.nonzero(real_atoms, as_tuple=True)

    block_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3), dtype=torch.float32, device=device
    )
    missing_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.int32, device=pbt.device
    )
    block_coords[real_atoms] = coords[
        nz_real_pose_ind, nz_real_block_ind, canonical_atom_inds[real_atoms]
    ]
    missing_atoms[real_atoms] = (
        atom_is_present[
            nz_real_pose_ind, nz_real_block_ind, canonical_atom_inds[real_atoms]
        ]
        == 0
    ).to(torch.int32)

    return block_coords, missing_atoms, real_atoms


@validate_args
def _annotate_packed_block_types_w_canonical_res_order(pbt: PackedBlockTypes):
    # mapping from canonical restype index to the
    # packed-block-types index for that restype
    #
    # what we want here is to be able to take four pieces of data:
    #  a. restype by 3-letter code
    #  b. termini assignment
    #  c. disulfide assignment
    #  d. his-d vs his-e assignment
    # and then look up into a tensor exactly which block type we are talking about

    if hasattr(pbt, "canonical_res_ordering_map"):
        assert hasattr(pbt, "bt_ind_to_canonical_ind")
        return

    max_n_termini_types = 3  # 0=Nterm, 1=mid, 2=Cterm
    max_n_aa_variant_types = 2  # CYS=0, CYD=1; HISE=0, HISD=1; all others, 0

    # forward ordering: from canonical-index + termini type + variant --> block-type index
    canonical_ordering_map = numpy.full(
        (len(ordered_canonical_aa_types), max_n_termini_types, max_n_aa_variant_types),
        -1,
        dtype=numpy.int32,
    )
    bt_ind_to_canonical_ind = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)

    def bt_inds_for_variant(base_names, var):
        return pbt.restype_index.get_indexer(
            [
                bt_name if var == "" else ":".join((bt_name, var))
                for bt_name in base_names
            ]
        )

    canonical_ordering_map[:, 0, 0] = bt_inds_for_variant(
        ordered_canonical_aa_types, "nterm"
    )
    canonical_ordering_map[:, 1, 0] = bt_inds_for_variant(
        ordered_canonical_aa_types, ""
    )
    canonical_ordering_map[:, 2, 0] = bt_inds_for_variant(
        ordered_canonical_aa_types, "cterm"
    )

    # NOTE: We have two amino acids with (non-termini) "variants" that we are going to handle
    # CYD, the disulfided CYS is variant 1; CYS is variant 0
    # between HIS and HIS_D, one of them is variant 1 and one is variant 0
    cys_his_var_names = [
        "CYD",
        "HIS_D" if his_taut_variant_ND1_protonated == 1 else "HIS",
    ]
    cys_his = (cys_co_aa_ind, his_co_aa_ind)
    canonical_ordering_map[cys_his, 0, 1] = bt_inds_for_variant(
        cys_his_var_names, "nterm"
    )
    canonical_ordering_map[cys_his, 1, 1] = bt_inds_for_variant(cys_his_var_names, "")
    canonical_ordering_map[cys_his, 2, 1] = bt_inds_for_variant(
        cys_his_var_names, "cterm"
    )

    # map from the specific block type to the generic canoncial aa index
    for i in range(len(canonical_ordering_map)):
        for j in range(3):
            for k in range(2):
                if canonical_ordering_map[i, j, k] != -1:
                    bt_ind_to_canonical_ind[canonical_ordering_map[i, j, k]] = i

    def t(x):
        return torch.tensor(x, dtype=torch.int64, device=pbt.device)

    setattr(pbt, "canonical_res_ordering_map", t(canonical_ordering_map))
    setattr(pbt, "bt_ind_to_canonical_ind", t(bt_ind_to_canonical_ind))


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


@validate_args
def _annotate_packed_block_types_w_canonical_atom_order(pbt: PackedBlockTypes):
    if hasattr(pbt, "canonical_atom_ind_map"):
        return
    canonical_atom_ind = numpy.full(
        (pbt.n_types, pbt.max_n_atoms), -1, dtype=numpy.int64
    )
    # canonical_res_ordering_map = pbt.canonical_res_ordering_map.cpu()
    for i, bt in enumerate(pbt.active_block_types):
        canonical_res_ind = pbt.bt_ind_to_canonical_ind[i]
        if canonical_res_ind == -1:
            continue
        canonical_res_name = ordered_canonical_aa_types[canonical_res_ind]
        # NOTE: We will use V2 for now, but this should be phased out
        i_canonical_ordering = ordered_canonical_aa_atoms_v2[canonical_res_name]
        for j, at in enumerate(bt.atoms):
            # probably this would be faster if we used a pandas indexer
            # but this is done only once, so, for now, use the slow form
            canonical_atom_ind[i, j] = i_canonical_ordering.index(at.name.strip())
    canonical_atom_ind = torch.tensor(
        canonical_atom_ind, dtype=torch.int64, device=pbt.device
    )
    setattr(pbt, "canonical_atom_ind_map", canonical_atom_ind)
