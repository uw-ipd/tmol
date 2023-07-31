import numpy
import torch

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
    chain_begin: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
    found_disulfides: Tensor[torch.int32][:, 3],
):
    pbt = packed_block_types
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    PoseStackBuilder._annotate_pbt_w_polymeric_down_up_bondsep_dist(pbt)
    PoseStackBuilder._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)

    device = pbt.device

    # canonica_res_ordering_map dimensioned: [20aas x 3 termini types x 2 variant types]
    # 3 termini types? 0-nterm, 1-mid, 2-cterm
    canonical_res_ordering_map = pbt.canonical_res_ordering_map
    res_types64 = res_types.to(torch.int64)
    res_type_variants64 = res_type_variants.to(torch.int64)
    real_res = res_types64 != -1

    assert torch.all(
        real_res[chain_begin == 1]
    ), "every residue marked as the beginning of a chain must be real"

    # TEMP! treat everything as a "mid" (1) termini type
    block_type_ind64 = torch.full_like(res_types64, -1)
    block_type_ind64[real_res] = canonical_res_ordering_map[
        res_types64[real_res], 1, res_type_variants64[real_res]
    ]
    # print("block_type_ind64[21]", block_type_ind64[0, 21])
    # print("res_types64[21]", res_types64[0, 21])
    # print("res_type_variants64[21]", res_type_variants64[0, 21])
    # print("canonical_res_ordering_map",
    #       canonical_res_ordering_map[
    #           res_types64[0, 21], 1, :
    #       ]
    #       )

    # block_type_ind64 = block_type_ind.to(torch.int64)

    # UGH: stealing/duplicating a lot of code from pose_stack_builder below
    n_poses = chain_begin.shape[0]
    max_n_res = chain_begin.shape[1]
    max_n_conn = pbt.max_n_conn
    inter_residue_connections64 = torch.full(
        (n_poses, max_n_res, max_n_conn, 2), -1, dtype=torch.int64, device=device
    )
    res_is_real_and_not_n_term = real_res.clone()
    res_is_real_and_not_n_term[chain_begin == 1] = False

    res_is_real_and_not_c_term = real_res.clone()
    n_pose_arange = torch.arange(n_poses, dtype=torch.int64, device=device)
    n_res = torch.sum(real_res, dim=1)
    res_is_real_and_not_c_term[n_pose_arange, n_res - 1] = False
    chain_end = torch.cat(
        (
            chain_begin[:, 1:],
            torch.zeros((n_poses, 1), dtype=torch.int32, device=device),
        ),
        dim=1,
    )
    chain_end[n_pose_arange, n_res - 1] = 1
    assert torch.all(
        real_res[chain_end == 1]
    ), "every residue marked as the end of a chain must be real"
    res_is_real_and_not_c_term[chain_end == 1] = False

    connected_up_conn_inds = pbt.up_conn_inds[
        block_type_ind64[res_is_real_and_not_c_term]
    ].to(torch.int64)
    connected_down_conn_inds = pbt.down_conn_inds[
        block_type_ind64[res_is_real_and_not_n_term]
    ].to(torch.int64)

    (
        nz_res_is_real_and_not_n_term_pose_ind,
        nz_res_is_real_and_not_n_term_res_ind,
    ) = torch.nonzero(res_is_real_and_not_n_term, as_tuple=True)
    (
        nz_res_is_real_and_not_c_term_pose_ind,
        nz_res_is_real_and_not_c_term_res_ind,
    ) = torch.nonzero(res_is_real_and_not_c_term, as_tuple=True)

    inter_residue_connections64[
        nz_res_is_real_and_not_c_term_pose_ind,
        nz_res_is_real_and_not_c_term_res_ind,
        connected_up_conn_inds,
        0,  # residue id
    ] = nz_res_is_real_and_not_n_term_res_ind
    inter_residue_connections64[
        nz_res_is_real_and_not_c_term_pose_ind,
        nz_res_is_real_and_not_c_term_res_ind,
        connected_up_conn_inds,
        1,  # connection id
    ] = connected_down_conn_inds

    inter_residue_connections64[
        nz_res_is_real_and_not_n_term_pose_ind,
        nz_res_is_real_and_not_n_term_res_ind,
        connected_down_conn_inds,
        0,  # residue id
    ] = nz_res_is_real_and_not_c_term_res_ind
    inter_residue_connections64[
        nz_res_is_real_and_not_n_term_pose_ind,
        nz_res_is_real_and_not_n_term_res_ind,
        connected_down_conn_inds,
        1,  # connection id
    ] = connected_up_conn_inds

    if found_disulfides.shape[0] != 0:
        found_disulfides64 = found_disulfides.to(torch.int64)
        cyd1_block_type64 = block_type_ind64[
            found_disulfides64[:, 0], found_disulfides64[:, 1]
        ]
        cyd2_block_type64 = block_type_ind64[
            found_disulfides64[:, 0], found_disulfides64[:, 2]
        ]
        # print("cyd1_block_type64:", cyd1_block_type64)
        # print("cyd2_block_type64:", cyd2_block_type64)
        # print([pbt.active_block_types[x].name for x in cyd1_block_type64])
        # print([pbt.active_block_types[x].name for x in cyd2_block_type64])

        # n- and c-term cyd residues will have different dslf connection inds
        # than mid-cyd residues; don't just hard code "2" here
        cyd1_dslf_conn64 = pbt.canonical_dslf_conn_ind[cyd1_block_type64].to(
            torch.int64
        )
        cyd2_dslf_conn64 = pbt.canonical_dslf_conn_ind[cyd2_block_type64].to(
            torch.int64
        )

        # print("cyd1_dslf_conn64")
        # print(cyd1_dslf_conn64)
        # print("cyd2_dslf_conn64")
        # print(cyd2_dslf_conn64)

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

    # now that we have the inter-residue connections sorted,
    # proceed with the rest of the PoseStackBuilder's steps
    # in constructing the inter_block_bondsep tensor
    # 3a
    (
        pconn_matrix,
        pconn_offsets,
        block_n_conn,
        pose_n_pconn,
    ) = PoseStackBuilder._take_real_conn_conn_intrablock_pairs(
        pbt, block_type_ind64, real_res
    )

    # 3b
    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        inter_residue_connections64, pconn_offsets, pconn_matrix
    )

    # 4
    ibb64 = PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph(
        pbt, block_n_conn, pose_n_pconn, pconn_matrix
    )
    inter_block_bondsep64 = ibb64

    return block_type_ind64, inter_residue_connections64, inter_block_bondsep64


@validate_args
def take_block_type_atoms_from_canonical(
    packed_block_types: PackedBlockTypes,
    chain_begin: Tensor[torch.int32][:, :],
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

    # block_types64 = block_types.to(torch.int64)

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
    # TEMP! do everything in numpy for the moment
    # TO DO! Use torch tensors on the device!
    #
    # mapping from canonical restype index to the
    # packed-block-types index for that restype
    #
    # what we want here is to be able to take four pieces of data:
    #  a. restype by 3-letter code
    #  b. termini assignment
    #  c. disulfide assignment
    #  d. his-d vs his-e assignment
    # and then look up into a tensor exactly which block type we are talking about
    #

    if hasattr(pbt, "canonical_res_ordering_map"):
        assert hasattr(pbt, "bt_ind_to_canonical_ind")
        return

    max_n_termini_types = 3  # TEMP! 0=Nterm, 1=mid, 2=Cterm
    max_n_aa_variant_types = 2  # TEMP! CYS=0, CYD=1; HISE=0, HISD=1; all others, 0

    # forward ordering: from canonical-index + termini type + variant --> block-type index
    canonical_ordering_map = numpy.full(
        (len(ordered_canonical_aa_types), max_n_termini_types, max_n_aa_variant_types),
        -1,
        dtype=numpy.int32,
    )
    bt_ind_to_canonical_ind = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)

    # TO DO: handle N- and C-termini variants
    var0_inds = pbt.restype_index.get_indexer(ordered_canonical_aa_types)
    canonical_ordering_map[:, 1, 0] = var0_inds

    # NOTE: We have two amino acids with (non-termini) "variants" that we are going to handle
    # CYD, the disulfided CYS is variant 1; CYS is variant 0
    # between HIS and HIS_D, one of them is variant 1 and one is variant 0
    canonical_ordering_map[
        (cys_co_aa_ind, his_co_aa_ind), 1, 1
    ] = pbt.restype_index.get_indexer(
        ["CYD", "HIS_D" if his_taut_variant_ND1_protonated == 1 else "HIS"]
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
        # print("bt", bt.name, "dslf?", "dslf" in bt.connection_to_cidx, bt.connection_to_cidx)
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
        # print("bt name:", bt.name)
        canonical_res_ind = pbt.bt_ind_to_canonical_ind[i]
        if canonical_res_ind == -1:
            continue
        canonical_res_name = ordered_canonical_aa_types[canonical_res_ind]
        ##### TEEEEEEMP!!!!!! #####
        # Use V2 for now
        i_canonical_ordering = ordered_canonical_aa_atoms_v2[canonical_res_name]
        for j, at in enumerate(bt.atoms):
            # probably this would be faster if we used a pandas indexer
            # but this is done only once, so, for now, use the slow form
            # print("at", at.name.strip())
            canonical_atom_ind[i, j] = i_canonical_ordering.index(at.name.strip())
    canonical_atom_ind = torch.tensor(
        canonical_atom_ind, dtype=torch.int64, device=pbt.device
    )
    setattr(pbt, "canonical_atom_ind_map", canonical_atom_ind)
