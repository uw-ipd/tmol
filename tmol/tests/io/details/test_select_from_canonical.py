import numpy
import torch
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.details.canonical_packed_block_types import (
    default_canonical_packed_block_types,
)
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
from tmol.io.details.select_from_canonical import (
    assign_block_types,
    take_block_type_atoms_from_canonical,
)
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_assign_block_types(torch_device, ubq_pdb):
    pbt, atr = default_canonical_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    ch_id, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_id = torch.tensor(ch_id, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    # 2
    found_disulfides, res_type_variants = find_disulfides(can_rts, coords, at_is_pres)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(pbt, ch_id, can_rts, res_type_variants, found_disulfides)

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


def test_take_block_type_atoms_from_canonical(torch_device, ubq_pdb):
    pbt, atr = default_canonical_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    ch_id, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_id = torch.tensor(ch_id, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    # 2
    found_disulfides, res_type_variants = find_disulfides(can_rts, coords, at_is_pres)
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(can_rts, res_type_variants, coords, at_is_pres)

    # now we'll invoke assign_block_types
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(pbt, ch_id, can_rts, res_type_variants, found_disulfides)

    block_coords, missing_atoms, real_atoms = take_block_type_atoms_from_canonical(
        pbt, block_types64, coords, at_is_pres
    )

    assert block_coords.device == torch_device
    assert missing_atoms.device == torch_device
    assert block_types64.device == torch_device

    assert block_coords.shape == (1, can_rts.shape[1], pbt.max_n_atoms, 3)
    assert missing_atoms.shape == block_coords.shape[:3]
    assert real_atoms.shape == missing_atoms.shape

    block_coords = block_coords.cpu().numpy()

    # all atoms are present in this weird PDB where Nterm
    # has H instead of 1H, 2H, & 3H,
    real_missing = torch.logical_and(missing_atoms, real_atoms)
    nz_rm_p, nz_rm_r, nz_rm_at = torch.nonzero(real_missing, as_tuple=True)
    for i in range(nz_rm_p.shape[0]):
        bt_i_ind = block_types64[0, nz_rm_r[i]]
        bt_i = pbt.active_block_types[bt_i_ind]
        print("atom", bt_i.atoms[nz_rm_at[i]].name, "missing from res", nz_rm_r[i])
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
    set_gold_coord(" 1H ", 26.961, 23.619, 2.168)
    set_gold_coord(" 2H ", 28.043, 24.834, 2.029)
    set_gold_coord(" 3H ", 27.746, 24.169, 3.490)
    set_gold_coord("  HA", 25.864, 25.717, 1.875)
    set_gold_coord(" 1HB", 24.227, 25.486, 3.461)
    set_gold_coord(" 2HB", 24.886, 23.861, 3.332)
    set_gold_coord(" 1HG", 26.298, 24.359, 5.342)
    set_gold_coord(" 2HG", 25.421, 25.882, 5.505)
    set_gold_coord(" 1HE", 23.700, 23.479, 8.233)
    set_gold_coord(" 2HE", 25.405, 23.472, 7.719)
    set_gold_coord(" 3HE", 24.552, 25.017, 7.954)

    numpy.testing.assert_equal(block_coords[0, 0], block_coords_res1_gold)
