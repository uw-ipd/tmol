import numpy
import torch

import cattr
import yaml
from attrs import evolve
from toolz.curried import groupby
from tmol.database.chemical import ChemicalDatabase, VariantType
from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.io.canonical_ordering import (
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
    CanonicalOrdering,
)
from tmol.io.details.left_justify_canonical_form import left_justify_canonical_form
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
from tmol.io.details.select_from_canonical import (
    assign_block_types,
    take_block_type_atoms_from_canonical,
)
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pose.packed_block_types import PackedBlockTypes


def not_any_nancoord(coords):
    return torch.logical_not(torch.any(torch.isnan(coords), dim=3))


def dslf_and_his_resolved_pose_stack_from_canonical_form(
    co, pbt, ch_id, can_rts, coords, at_is_pres
):
    ch_id, can_rts, coords, at_is_pres, _1, _2 = left_justify_canonical_form(
        ch_id, can_rts, coords, at_is_pres
    )

    # 2
    found_disulfides, res_type_variants = find_disulfides(co, can_rts, coords)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(co, can_rts, res_type_variants, coords, at_is_pres)

    return (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    )


def test_assign_block_types(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)
    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    # ubq seq
    ubq_1lc = [
        x
        for x in "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    ubq_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(ubq_1lc)
    ubq_bt_inds = numpy.expand_dims(
        pbt.bt_mapping_w_lcaa_1lc.iloc[ubq_df_inds]["bt_ind"].values, axis=0
    )
    ubq_bt_inds[0, 0] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "MET:nterm"
    )
    ubq_bt_inds[0, -1] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "GLY:cterm"
    )

    assert block_types.device == torch_device
    assert inter_residue_connections64.device == torch_device
    assert inter_residue_connections64.dtype == torch.int64

    numpy.testing.assert_equal(block_types.cpu().numpy(), ubq_bt_inds)


def test_assign_block_types_w_exotic_termini_options(
    default_database, ubq_pdb, torch_device
):
    floro_nterm_patch = """
  - name:  AminoFloroTerminus
    display_name: floro_nterm
    pattern: '[*][*][NH][{down}]'
    remove_atoms:
    - <{down}>
    - <H1>
    add_atoms:
    - { name: F1  ,  atom_type: S }
    - { name: F2  ,  atom_type: S }
    - { name: F3  ,  atom_type: S }
    add_atom_aliases:
    - { name: F1  ,  alt_name: 1F }
    - { name: F2  ,  alt_name: 2F }
    - { name: F3  ,  alt_name: 3F }

    modify_atoms:
    - { name: <N1> ,  atom_type: Nlys }
    add_connections: []
    add_bonds:
    - [  <N1>,    F1   ]
    - [  <N1>,    F2   ]
    - [  <N1>,    F3   ]
    icoors:
    - { name:   F1, source: <H1>, phi: 180.0 deg, theta: 70.5 deg, d: 1.35, parent:  <N1>, grand_parent: <*2>, great_grand_parent: <*1> }
    - { name:   F2, source: <H1>, phi: 120.0 deg, theta: 70.5 deg, d: 1.35, parent:  <N1>, grand_parent: <*2>, great_grand_parent: F1 }
    - { name:   F3, source: <H1>, phi: 120.0 deg, theta: 70.5 deg, d: 1.35, parent:  <N1>, grand_parent: <*2>, great_grand_parent: F2 }
"""

    def variant_from_yaml(yml_string):
        raw = yaml.safe_load(yml_string)
        return tuple(cattr.structure(x, VariantType) for x in raw)

    floro_nterm_variant = variant_from_yaml(floro_nterm_patch)

    chem_db = default_database.chemical
    chem_elem_types = chem_db.element_types
    chem_atom_types = chem_db.atom_types

    unpatched_res = [res for res in chem_db.residues if res.name == res.base_name]
    ext_chemical_db = ChemicalDatabase(
        element_types=chem_elem_types,
        atom_types=chem_atom_types,
        residues=unpatched_res,
        variants=(chem_db.variants + floro_nterm_variant),
    )
    patched_chem_db = PatchedChemicalDatabase.from_chem_db(ext_chemical_db)

    co = CanonicalOrdering.from_chemdb(patched_chem_db)
    restype_list = [
        cattr.structure(
            cattr.unstructure(r),
            RefinedResidueType,
        )
        for r in patched_chem_db.residues
    ]

    restype_map = groupby(lambda restype: restype.name3, restype_list)
    restype_set = ResidueTypeSet(
        residue_types=restype_list,
        restype_map=restype_map,
        chem_db=patched_chem_db,
    )

    pbt = PackedBlockTypes.from_restype_list(
        patched_chem_db, restype_set, restype_list, torch_device
    )

    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)
    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    # ubq seq
    ubq_1lc = [
        x
        for x in "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    ubq_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(ubq_1lc)
    ubq_bt_inds = numpy.expand_dims(
        pbt.bt_mapping_w_lcaa_1lc.iloc[ubq_df_inds]["bt_ind"].values, axis=0
    )
    ubq_bt_inds[0, 0] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "MET:nterm"
    )
    ubq_bt_inds[0, -1] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "GLY:cterm"
    )

    assert block_types.device == torch_device
    assert inter_residue_connections64.device == torch_device
    assert inter_residue_connections64.dtype == torch.int64

    numpy.testing.assert_equal(block_types.cpu().numpy(), ubq_bt_inds)


def test_assign_block_types_jagged_poses(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    # first 4 res
    cf4 = canonical_form_from_pdb(co, ubq_pdb, torch_device, residue_end=4)
    ch_id_4, can_rts_4, coords_4 = cf4["chain_id"], cf4["res_types"], cf4["coords"]
    at_is_pres_4 = not_any_nancoord(coords_4)
    # first 6 res
    cf6 = canonical_form_from_pdb(co, ubq_pdb, torch_device, residue_end=6)
    ch_id_6, can_rts_6, coords_6 = cf6["chain_id"], cf6["res_types"], cf6["coords"]
    at_is_pres_6 = not_any_nancoord(coords_6)

    def ext_one(x4, x6, fill_val):
        new_shape = (2,) + x6.shape[1:]
        x = torch.full(new_shape, fill_val, dtype=x4.dtype, device=x4.device)
        x[0, : x4.shape[1]] = x4
        x[1] = x6
        return x

    ch_id = ext_one(ch_id_4, ch_id_6, -1)
    can_rts = ext_one(can_rts_4, can_rts_6, -1)
    coords = ext_one(coords_4, coords_6, numpy.nan)
    at_is_pres = ext_one(at_is_pres_4, at_is_pres_6, False)

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    # ubq seq
    ubq_1lc = [
        x
        for x in "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    ubq_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(ubq_1lc)
    ubq_bt_inds = numpy.expand_dims(
        pbt.bt_mapping_w_lcaa_1lc.iloc[ubq_df_inds]["bt_ind"].values, axis=0
    )

    jagged_gold_bt_inds = numpy.full((2, 6), -1, dtype=numpy.int64)
    jagged_gold_bt_inds[0, :4] = ubq_bt_inds[0, :4]
    jagged_gold_bt_inds[0, 0] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "MET:nterm"
    )
    jagged_gold_bt_inds[0, 3] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "PHE:cterm"
    )
    jagged_gold_bt_inds[1, :6] = ubq_bt_inds[0, :6]
    jagged_gold_bt_inds[1, 0] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "MET:nterm"
    )
    jagged_gold_bt_inds[1, 5] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "LYS:cterm"
    )

    assert block_types.device == torch_device
    assert inter_residue_connections64.device == torch_device
    assert inter_residue_connections64.dtype == torch.int64

    numpy.testing.assert_equal(block_types.cpu().numpy(), jagged_gold_bt_inds)


def test_assign_block_types_with_gaps(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    # take ten residues
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device, residue_end=10)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)

    # put two empty residues in between res 5 and 6
    def add_two_res(x, fill_value):
        if len(x.shape) >= 3:
            fill_shape = (x.shape[0], 2, *x.shape[2:])
        else:
            fill_shape = (x.shape[0], 2)
        return torch.cat(
            [
                x[:, :5],
                torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device),
                x[:, 5:],
            ],
            dim=1,
        )

    ch_id = add_two_res(ch_id, 0)
    can_rts = add_two_res(can_rts, -1)
    coords = add_two_res(coords, float("nan"))
    at_is_pres = add_two_res(at_is_pres, 0)

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    inter_res_conn_gold = numpy.full((1, 12, 3, 2), -1, dtype=numpy.int64)

    def p(x, y):
        return numpy.array([x, y], dtype=numpy.int64)

    inter_res_conn_gold[0, 0, 0, :] = p(1, 0)
    inter_res_conn_gold[0, 1, 0, :] = p(0, 0)
    inter_res_conn_gold[0, 1, 1, :] = p(2, 0)
    inter_res_conn_gold[0, 2, 0, :] = p(1, 1)
    inter_res_conn_gold[0, 2, 1, :] = p(3, 0)
    inter_res_conn_gold[0, 3, 0, :] = p(2, 1)
    inter_res_conn_gold[0, 3, 1, :] = p(4, 0)
    inter_res_conn_gold[0, 4, 0, :] = p(3, 1)
    inter_res_conn_gold[0, 4, 1, :] = p(5, 0)
    inter_res_conn_gold[0, 5, 0, :] = p(4, 1)
    inter_res_conn_gold[0, 5, 1, :] = p(6, 0)
    inter_res_conn_gold[0, 6, 0, :] = p(5, 1)
    inter_res_conn_gold[0, 6, 1, :] = p(7, 0)
    inter_res_conn_gold[0, 7, 0, :] = p(6, 1)
    inter_res_conn_gold[0, 7, 1, :] = p(8, 0)
    inter_res_conn_gold[0, 8, 0, :] = p(7, 1)
    inter_res_conn_gold[0, 8, 1, :] = p(9, 0)
    inter_res_conn_gold[0, 9, 0, :] = p(8, 1)

    numpy.testing.assert_equal(
        inter_res_conn_gold, inter_residue_connections64.cpu().numpy()
    )


def test_assign_block_types_with_same_chain_cterm_vrt(ubq_pdb, torch_device):
    # We should allow virtual residues and generally ligands to be labeled
    # as being part of a given chain, even if geometrically and chemically
    # that makes no sense; e.g., waters are often listed as being part of
    # a particular chain. What we want to ensure happens is that
    # the non-virtual C-terminal residue is still labeled as a C-term type.

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    # take ten residues
    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device, residue_end=10)

    # Now let's add a virtual residue to the end of the chain
    vrt_co_ind = co.restype_io_equiv_classes.index("VRT")
    # print("vrt_co_ind", vrt_co_ind)
    orig_coords = cf["coords"]
    ocs = orig_coords.shape
    new_coords = torch.full(
        (ocs[0], ocs[1] + 1, ocs[2], ocs[3]),
        numpy.nan,
        dtype=torch.float32,
        device=torch_device,
    )
    new_coords[0, :-1, :, :] = orig_coords

    # Let's put the VRT right in the center of ILE 3
    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32, device=torch_device)

    new_coords[0, -1, 0, :] = xyz(26.849, 29.656, 6.217)
    new_coords[0, -1, 1, :] = xyz(26.849, 29.656, 6.217) + xyz(1.0, 0.0, 0.0)
    new_coords[0, -1, 2, :] = xyz(26.849, 29.656, 6.217) + xyz(0.0, 1.0, 0.0)
    orig_chain_id = cf["chain_id"]

    ocis = orig_chain_id.shape
    new_chain_id = torch.zeros(
        (ocis[0], ocis[1] + 1), dtype=torch.int32, device=torch_device
    )
    new_chain_id[0, :-1] = orig_chain_id
    new_chain_id[0, -1] = orig_chain_id[
        0, -1
    ]  # give the vrt res the same chain id as the last residue

    orig_restypes = cf["res_types"]
    ors = orig_restypes.shape
    new_restypes = torch.full(
        (ors[0], ors[1] + 1), -1, dtype=torch.int32, device=torch_device
    )
    new_restypes[0, :-1] = orig_restypes
    new_restypes[0, -1] = vrt_co_ind

    cf["coords"] = new_coords
    cf["chain_id"] = new_chain_id
    cf["res_types"] = new_restypes

    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)

    # # put two empty residues in between res 5 and 6
    # def add_two_res(x, fill_value):
    #     if len(x.shape) >= 3:
    #         fill_shape = (x.shape[0], 2, *x.shape[2:])
    #     else:
    #         fill_shape = (x.shape[0], 2)
    #     return torch.cat(
    #         [
    #             x[:, :5],
    #             torch.full(fill_shape, fill_value, dtype=x.dtype, device=x.device),
    #             x[:, 5:],
    #         ],
    #         dim=1,
    #     )

    # ch_id = add_two_res(ch_id, 0)
    # can_rts = add_two_res(can_rts, -1)
    # coords = add_two_res(coords, float("nan"))
    # at_is_pres = add_two_res(at_is_pres, 0)

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    penultimate_res_block_type = block_types[0, -2]
    penultimate_bt = pbt.active_block_types[penultimate_res_block_type]
    assert penultimate_bt.name.partition(":")[2] == "cterm"


def test_assign_block_types_for_pert_and_antigen(
    pertuzumab_and_nearby_erbb2_pdb_and_segments, torch_device
):
    co = default_canonical_ordering()
    (
        pert_and_erbb2_lines,
        res_not_connected,
    ) = pertuzumab_and_nearby_erbb2_pdb_and_segments

    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    cf = canonical_form_from_pdb(co, pert_and_erbb2_lines, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)
    res_not_connected = torch.tensor(res_not_connected, device=torch_device)

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )
    # skip dslf and his-taut steps

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co,
        pbt,
        at_is_pres,
        ch_id,
        can_rts,
        res_type_variants,
        found_disulfides,
        res_not_connected,
    )

    assert block_types.device == torch_device
    assert inter_residue_connections64.device == torch_device
    assert inter_residue_connections64.dtype == torch.int64

    block_types = block_types.cpu().numpy()
    inter_residue_connections64 = inter_residue_connections64.cpu().numpy()

    chain1_last_res = 214 - 1
    chain2_last_res = chain1_last_res + 222

    for i, bt_ind in enumerate(block_types[0, :]):
        bt = pbt.active_block_types[bt_ind]
        if i == 0 or i == chain1_last_res + 1:
            assert bt.name.partition(":")[2] == "nterm"
            # cterm connection index is 0 for nterm-patched types
            assert inter_residue_connections64[0, i, 0, 0] == i + 1
            assert inter_residue_connections64[0, i, 0, 1] == 0
        elif i == chain1_last_res or i == chain2_last_res:
            assert bt.name.partition(":")[2] == "cterm"
            assert inter_residue_connections64[0, i, 0, 0] == i - 1
            assert inter_residue_connections64[0, i, 0, 1] == 1
        else:
            assert bt.name.partition(":")[2] == ""
            if res_not_connected[0, i, 0]:
                assert inter_residue_connections64[0, i, 0, 0] == -1
                assert inter_residue_connections64[0, i, 0, 1] == -1
                assert inter_residue_connections64[0, i, 1, 0] == i + 1
                assert inter_residue_connections64[0, i, 1, 1] == 0
            elif res_not_connected[0, i, 1]:
                assert inter_residue_connections64[0, i, 0, 0] == i - 1
                assert inter_residue_connections64[0, i, 0, 1] == 1
                assert inter_residue_connections64[0, i, 1, 0] == -1
                assert inter_residue_connections64[0, i, 1, 1] == -1
            else:
                assert inter_residue_connections64[0, i, 0, 0] == i - 1
                if i - 1 == 0 or i - 1 == chain1_last_res + 1:
                    # previous res is nterm
                    assert inter_residue_connections64[0, i, 0, 1] == 0
                else:
                    assert inter_residue_connections64[0, i, 0, 1] == 1
                assert inter_residue_connections64[0, i, 1, 0] == i + 1
                assert inter_residue_connections64[0, i, 1, 1] == 0


def test_take_block_type_atoms_from_canonical(torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)
    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    (
        block_coords,
        missing_atoms,
        real_atoms,
        real_canonical_atom_inds,
    ) = take_block_type_atoms_from_canonical(pbt, block_types64, coords, at_is_pres)

    assert block_coords.device == torch_device
    assert missing_atoms.device == torch_device
    assert block_types64.device == torch_device
    assert real_canonical_atom_inds.device == torch_device

    assert block_coords.shape == (1, can_rts.shape[1], pbt.max_n_atoms, 3)
    assert missing_atoms.shape == block_coords.shape[:3]
    assert real_atoms.shape == missing_atoms.shape
    assert real_canonical_atom_inds.dtype == torch.int64

    block_coords = block_coords.cpu().numpy()

    # all atoms are present in this weird PDB where Nterm
    # has H instead of 1H, 2H, & 3H,
    real_missing = torch.logical_and(missing_atoms, real_atoms)
    nz_rm_p, nz_rm_r, nz_rm_at = torch.nonzero(real_missing, as_tuple=True)
    assert torch.sum(torch.logical_and(missing_atoms, real_atoms)).item() == 0

    # ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
    # ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
    # ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
    # ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O
    # ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 13.77           C
    # ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 16.29           C
    # ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 17.17           S
    # ATOM      8  CE  MET A   1      24.447  23.984   7.620  1.00 16.11           C
    # ATOM      9  H   MET A   1      27.282  23.521   3.027  1.00 11.60           H
    # ATOM     10  HA  MET A   1      25.864  25.717   1.875  1.00 12.46           H
    # ATOM     11 1HB  MET A   1      24.227  25.486   3.461  1.00 16.52           H
    # ATOM     12 2HB  MET A   1      24.886  23.861   3.332  1.00 16.52           H
    # ATOM     13 1HG  MET A   1      26.298  24.359   5.342  1.00 19.55           H
    # ATOM     14 2HG  MET A   1      25.421  25.882   5.505  1.00 19.55           H
    # ATOM     15 1HE  MET A   1      23.700  23.479   8.233  1.00 19.33           H
    # ATOM     16 2HE  MET A   1      25.405  23.472   7.719  1.00 19.33           H
    # ATOM     17 3HE  MET A   1      24.552  25.017   7.954  1.00 19.33           H

    block_coords_res1_gold = numpy.zeros((pbt.max_n_atoms, 3), dtype=numpy.float32)
    met_bt = next(x for x in pbt.active_block_types if x.name == "MET:nterm")

    def set_gold_coord(name, x, y, z):
        ind = next(i for i, at in enumerate(met_bt.atoms) if at.name == name.strip())
        block_coords_res1_gold[ind, 0] = x
        block_coords_res1_gold[ind, 1] = y
        block_coords_res1_gold[ind, 2] = z

    set_gold_coord("  N ", 27.340, 24.430, 2.614)
    set_gold_coord("  CA", 26.266, 25.413, 2.842)
    set_gold_coord("  C ", 26.913, 26.639, 3.531)
    set_gold_coord("  O ", 27.886, 26.463, 4.263)
    set_gold_coord("  CB", 25.112, 24.880, 3.649)
    set_gold_coord("  CG", 25.353, 24.860, 5.134)
    set_gold_coord("  SD", 23.930, 23.959, 5.904)
    set_gold_coord("  CE", 24.447, 23.984, 7.620)
    set_gold_coord("  H1", 26.961, 23.619, 2.168)
    set_gold_coord("  H2", 28.043, 24.834, 2.029)
    set_gold_coord("  H3", 27.746, 24.169, 3.490)
    set_gold_coord("  HA", 25.864, 25.717, 1.875)
    set_gold_coord(" HB2", 24.227, 25.486, 3.461)
    set_gold_coord(" HB3", 24.886, 23.861, 3.332)
    set_gold_coord(" HG2", 26.298, 24.359, 5.342)
    set_gold_coord(" HG3", 25.421, 25.882, 5.505)
    set_gold_coord(" HE1", 23.700, 23.479, 8.233)
    set_gold_coord(" HE2", 25.405, 23.472, 7.719)
    set_gold_coord(" HE3", 24.552, 25.017, 7.954)

    numpy.testing.assert_equal(block_coords[0, 0], block_coords_res1_gold)


def variants_from_yaml(yml_string):
    raw = yaml.safe_load(yml_string)
    return tuple(cattr.structure(x, VariantType) for x in raw)


def co_and_pbt_from_new_variants(ducd, patches, device):
    new_ucd = evolve(ducd, variants=(ducd.variants + variants_from_yaml(patches)))

    new_pucd = PatchedChemicalDatabase.from_chem_db(new_ucd)

    co = CanonicalOrdering.from_chemdb(new_pucd)
    restype_list = [
        cattr.structure(cattr.unstructure(rt), RefinedResidueType)
        for rt in new_pucd.residues
    ]
    restype_map = groupby(lambda restype: restype.name3, restype_list)

    restype_set = ResidueTypeSet(
        residue_types=restype_list,
        restype_map=restype_map,
        chem_db=new_pucd,
    )

    pbt = PackedBlockTypes.from_restype_list(
        new_pucd, restype_set, restype_list, device
    )

    return co, pbt, new_pucd


def test_select_best_block_type_candidate_choosing_default_term(
    torch_device, ubq_pdb, default_unpatched_chemical_database
):
    ducd = default_unpatched_chemical_database
    patch = """
  - name:  FakeCarboxyTerminus
    display_name: fake_cterm
    pattern: '[*][*]C(=O)[{up}]'
    remove_atoms:
    - <{up}>
    - <O1>
    add_atoms:
    - { name: O    ,  atom_type: OOC }
    - { name: OXT  ,  atom_type: OOC }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [   <C1>,    OXT ]
    - [   <C1>,    O ]
    icoors:
    - { name:     O, source: <O1>, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, parent:  <C1>, grand_parent: <*2>, great_grand_parent: <*1>}
    - { name:   OXT, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  <C1>, grand_parent: <*2>, great_grand_parent: O }
    """

    co, pbt, new_pucd = co_and_pbt_from_new_variants(ducd, patch, torch_device)

    varnames = [var.display_name for var in new_pucd.variants]
    assert "fake_cterm" in varnames

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    # let's look at the block type assigned to c-term
    bt_cterm_ind = block_types[0, -1].item()

    bt_cterm = pbt.active_block_types[bt_cterm_ind]
    assert bt_cterm.name == "GLY:cterm"


def pser_and_mser_patches():
    return """
  - name:  PhosphatePatch
    display_name: phospho
    pattern: '[*][*]C[OH]'
    remove_atoms:
    - <H1>
    add_atoms:
    - { name: P   ,  atom_type: OOC }
    - { name: OP1  ,  atom_type: OOC }
    - { name: OP2  ,  atom_type: OOC }
    - { name: OP3  ,  atom_type: OOC }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [   <O1>,   P ]
    - [   P,    OP1 ]
    - [   P,    OP2 ]
    - [   P,    OP3 ]
    icoors:
    - { name:     P, source: <H1>, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, parent:  <O1>, grand_parent: <C1>, great_grand_parent: <*1>}
    - { name:   OP1, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  P, grand_parent: <O1>, great_grand_parent: <*1> }
    - { name:   OP2, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  P, grand_parent: <O1>, great_grand_parent: <C1> }
    - { name:   OP3, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  P, grand_parent: <O1>, great_grand_parent: <C1> }

  - name:  MosphatePatch
    display_name: mospho
    pattern: '[*][*]C[OH]'
    remove_atoms:
    - <H1>
    add_atoms:
    - { name: M   ,  atom_type: OOC }
    - { name: OM1  ,  atom_type: OOC }
    - { name: OM2  ,  atom_type: OOC }
    - { name: OM3  ,  atom_type: OOC }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [   <O1>,   M ]
    - [   M,    OM1 ]
    - [   M,    OM2 ]
    - [   M,    OM3 ]
    icoors:
    - { name:     M, source: <H1>, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, parent:  <O1>, grand_parent: <C1>, great_grand_parent: <*1>}
    - { name:   OM1, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  M, grand_parent: <O1>, great_grand_parent: <*1> }
    - { name:   OM2, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  M, grand_parent: <O1>, great_grand_parent: <C1> }
    - { name:   OM3, source: <O1>, phi: -180.0 deg, theta: 60.0 deg, d: 1.2, parent:  M, grand_parent: <O1>, great_grand_parent: <C1> }
    """


def test_select_best_block_type_candidate_w_mult_opts(
    torch_device, ubq_pdb, default_unpatched_chemical_database
):
    ducd = default_unpatched_chemical_database

    # two patches that do the job of adding atoms to the SER/THR
    # hydroxyls but that do not do the job of defining good
    # chemistry or geometry; that's not their jobs
    patch = pser_and_mser_patches()
    co, pbt, new_pucd = co_and_pbt_from_new_variants(ducd, patch, torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    co_ser_atom_ind_map = co.restypes_atom_index_mapping["SER"]
    co_ser_HG_ind = co_ser_atom_ind_map["HG"]
    co_mser_M_ind = co_ser_atom_ind_map["M"]

    varnames = [var.display_name for var in new_pucd.variants]
    bt_names = [bt.name for bt in new_pucd.residues]

    assert "phospho" in varnames
    assert "mospho" in varnames
    assert "SER:phospho" in bt_names
    assert "SER:mospho" in bt_names
    assert "THR:phospho" in bt_names
    assert "THR:mospho" in bt_names

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)

    at_is_pres[0, 19, co_mser_M_ind] = True
    at_is_pres[0, 19, co_ser_HG_ind] = False
    coords[0, 19, co_mser_M_ind] = 0

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
    )

    # ubq seq
    ubq_1lc = [
        x
        for x in "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    ubq_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(ubq_1lc)
    ubq_bt_inds = numpy.expand_dims(
        pbt.bt_mapping_w_lcaa_1lc.iloc[ubq_df_inds]["bt_ind"].values, axis=0
    )
    ubq_bt_inds[0, 0] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "MET:nterm"
    )
    ubq_bt_inds[0, -1] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "GLY:cterm"
    )
    ubq_bt_inds[0, 19] = next(
        i for i, bt in enumerate(pbt.active_block_types) if bt.name == "SER:mospho"
    )
    numpy.testing.assert_equal(block_types.cpu().numpy(), ubq_bt_inds)


def test_select_best_block_type_candidate_error_impossible_combo(
    torch_device, ubq_pdb, default_unpatched_chemical_database
):
    ducd = default_unpatched_chemical_database

    # two patches that do the job of adding atoms to the SER/THR
    # hydroxyls but that do not do the job of defining good
    # chemistry or geometry; that's not their jobs
    patch = pser_and_mser_patches()
    co, pbt, new_pucd = co_and_pbt_from_new_variants(ducd, patch, torch_device)

    co_ser_atom_ind_map = co.restypes_atom_index_mapping["SER"]
    co_pser_P_ind = co_ser_atom_ind_map["P"]
    co_mser_M_ind = co_ser_atom_ind_map["M"]

    varnames = [var.display_name for var in new_pucd.variants]
    bt_names = [bt.name for bt in new_pucd.residues]

    assert "phospho" in varnames
    assert "mospho" in varnames
    assert "SER:phospho" in bt_names
    assert "SER:mospho" in bt_names
    assert "THR:phospho" in bt_names
    assert "THR:mospho" in bt_names

    cf = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    ch_id, can_rts, coords = cf["chain_id"], cf["res_types"], cf["coords"]
    at_is_pres = not_any_nancoord(coords)

    at_is_pres[0, 19, co_pser_P_ind] = True
    at_is_pres[0, 19, co_mser_M_ind] = True
    coords[0, 19, co_pser_P_ind] = 0
    coords[0, 19, co_mser_M_ind] = 0

    (
        ch_id,
        can_rts,
        coords,
        at_is_pres,
        found_disulfides,
        res_type_variants,
        his_taut,
        resolved_coords,
        resolved_atom_is_present,
    ) = dslf_and_his_resolved_pose_stack_from_canonical_form(
        co, pbt, ch_id, can_rts, coords, at_is_pres
    )

    expected_err_msg = """failed to resolve a block type from the candidates available
 Failed to resolve block type for 0 19 SER
 0 19 0 68 SER restype 15 equiv class SER
  atom P provided but absent from candidate SER
  atom M provided but absent from candidate SER
 Failed to resolve block type for 0 19 SER
 0 19 1 71 SER:phospho restype 15 equiv class SER
  atom HG provided but absent from candidate SER:phospho
  atom M provided but absent from candidate SER:phospho
 Failed to resolve block type for 0 19 SER
 0 19 2 72 SER:mospho restype 15 equiv class SER
  atom HG provided but absent from candidate SER:mospho
  atom P provided but absent from candidate SER:mospho
"""

    try:
        (
            block_types,
            inter_residue_connections64,
            inter_block_bondsep64,
        ) = assign_block_types(
            co, pbt, at_is_pres, ch_id, can_rts, res_type_variants, found_disulfides
        )
    except RuntimeError as err:
        assert str(err) == expected_err_msg
