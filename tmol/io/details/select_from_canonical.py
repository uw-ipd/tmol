import numpy
import torch

from typing import Optional, Tuple
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.io.canonical_ordering import (
    CanonicalOrdering,
    # ordered_canonical_aa_types,
    # ordered_canonical_aa_atoms_v2,
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
    canonical_ordering: CanonicalOrdering,
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
    _annotate_packed_block_types_w_canonical_res_order(canonical_ordering, pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(canonical_ordering, pbt)
    PoseStackBuilder._annotate_pbt_w_polymeric_down_up_bondsep_dist(pbt)
    PoseStackBuilder._annotate_pbt_w_intraresidue_connection_atom_distances(pbt)

    device = pbt.device
    n_poses = chain_id.shape[0]
    max_n_res = chain_id.shape[1]
    max_n_conn = pbt.max_n_conn

    # canonical_res_ordering_map dimensioned: [20aas x 3 termini types x 2 special-case variant types]
    # 3 termini types? 0-nterm, 1-mid, 2-cterm,
    canonical_res_ordering_map = pbt.canonical_res_ordering_map
    res_types64 = res_types.to(torch.int64)
    res_type_variants64 = res_type_variants.to(torch.int64)
    is_real_res = res_types64 != -1

    if res_not_connected is None:
        res_not_connected = torch.zeros(
            (n_poses, max_n_res, 2), dtype=torch.bool, device=device
        )

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
    # SHOULD THIS JUST GO IN POSE_STACK_BUILDER AND REPLACE ITS EXISTING CODE???
    # SHOULD POSE_STACK_BUILDER BE DEPRECATED??
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
    real_canonical_atom_inds = canonical_atom_inds[real_atoms]
    block_coords[real_atoms] = coords[
        nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds
    ]
    missing_atoms[real_atoms] = (
        atom_is_present[nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds]
        == 0
    ).to(torch.int32)

    return (block_coords, missing_atoms, real_atoms, real_canonical_atom_inds)


class CanonicalOrderingBlockTypeMap:
    def __init__(self, bt_ordering_map, bt_ind_to_canonical_ind):
        self.bt_ordering_map = bt_ordering_map
        self.bt_ind_to_canonical_ind = bt_ind_to_canonical_ind


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

    if (
        hasattr(pbt, "canonical_ording_annotation")
        and id(co) in pbt.canonical_ordering_annotation
    ):
        return

    # what is the problem we are trying to solve?
    # we have a number of "name3s" that we want to map
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

    max_n_defaultdict(int)
    base_blocktypes = {}
    for bt in pbt.active_block_types:
        if bt.name.find(":") == -1:
            assert bt.base_name == bt.name
            base_blocktypes[bt.base_name] = bt

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

    def term_and_spcase_variant_lists():
        variants = []
        for i in range(max_n_termini_types):
            variants.append([])
            for j in range(max_n_special_case_aa_variant_types):
                variants[i].append([])
        return variants

    base_name_set = set([bt.base_name for bt in pbt.active_block_types])
    default_type_variants = {
        base_name: term_and_spcase_variant_lists() for base_name in base_name_set
    }

    # the number of base types in this PBT, which may represent a subset
    # of the base types in the ChemicalDatabase from whith the CO was
    # derived
    n_base_types = len(base_name_set)

    for i, bt in enumerate(pbt.active_block_types):
        bt_vars = bt.name.split(":")
        var_is_down_term = False
        var_is_up_term = False
        var_is_non_default_term = False
        var_is_cyd = bt.base_name == "CYD"
        var_is_hisd = bt.base_name == "HIS_D"
        for var_type in bt_vars[1:]:
            if var_type in co.down_termini_variants:
                var_is_down_term = True
                if var_type != co.restypes_default_termini_mapping[bt.base_name][0]:
                    var_is_non_default_term = True
            if var_type in co.up_termini_variants:
                var_is_up_term = True
                if var_type != co.restypes_default_termini_mapping[bt.base_name][1]:
                    var_is_non_default_term = True
        term_ind = map_term_to_int(var_is_down_term, var_is_up_term)
        spcase_var_ind = map_spcase_var_to_int(var_is_cyd, var_is_hisd)
        base_type_variants[bt.base_name][term_ind][spcase_var_ind].append(bt)

    max_variants_for_fixed_term_and_spcase = 1
    for bt in base_type_variants:
        for i in range(max_n_termini_types):
            for j in range(max_n_special_case_aa_variant_types):
                n_vars = len(base_type_variants[bt][i][j])
                if n_vars > max_variants_for_fixed_term_and_spcase:
                    max_variants_for_fixed_term_and_spcase = n_vars

    # RESUME WORK HERE
    bt_var_atom_is_present = torch.zeros(
        (
            n_base_types,
            max_n_termini_types,
            max_n_special_case_aa_variant_types,
            max_variants_for_fixed_term_and_spcase,
            co.max_n_canonical_atoms,
        ),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )

    variants_for_blocktypes = defaultdict(list)
    non_default_termini_types = defaultdict(lambda: set([]))
    for bt in bt.active_block_types:
        variants_for_blocktypes[bt.base_name].append(bt)

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
