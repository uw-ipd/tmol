import numpy
import torch
from tmol.io.canonical_ordering import (
    canonical_form_from_pdb_lines,
)
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

    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
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
    ) = assign_block_types(pbt, ch_beg, can_rts, res_type_variants, found_disulfides)

    # ubq seq
    ubq_1lc = [
        x
        for x in "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    ubq_df_inds = pbt.bt_mapping_w_lcaa_1lc_ind.get_indexer(ubq_1lc)
    ubq_bt_inds = numpy.expand_dims(
        pbt.bt_mapping_w_lcaa_1lc.iloc[ubq_df_inds]["bt_ind"].values, axis=0
    )

    numpy.testing.assert_equal(block_types.cpu().numpy(), ubq_bt_inds)


def test_take_block_type_atoms_from_canonical(torch_device, ubq_pdb):
    pbt, atr = default_canonical_packed_block_types(torch_device)
    PoseStackBuilder._annotate_pbt_w_canonical_aa1lc_lookup(pbt)

    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
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
    ) = assign_block_types(pbt, ch_beg, can_rts, res_type_variants, found_disulfides)

    block_coords, missing_atoms, real_atoms = take_block_type_atoms_from_canonical(
        pbt,
        ch_beg,
        block_types64,
        coords,
        at_is_pres,
    )

    assert block_coords.device == torch_device
    assert missing_atoms.device == torch_device
    assert block_types64.device == torch_device

    n_ats = torch.sum(pbt.n_atoms[block_types64[0]])

    assert block_coords.shape == (1, can_rts.shape[1], pbt.max_n_atoms, 3)
    assert missing_atoms.shape == block_coords.shape[:3]
    assert real_atoms.shape == missing_atoms.shape

    # all atoms are present in this weird PDB where Nterm
    # has H instead of 1H, 2H, & 3H,
    real_missing = torch.logical_and(missing_atoms, real_atoms)
    nz_rm_p, nz_rm_r, nz_rm_at = torch.nonzero(real_missing, as_tuple=True)
    for i in range(nz_rm_p.shape[0]):
        bt_i_ind = block_types64[0, nz_rm_r[i]]
        bt_i = pbt.active_block_types[bt_i_ind]
        print("atom", bt_i.atoms[nz_rm_at[i]].name, "missing from res", nz_rm_r[i])
    assert torch.sum(torch.logical_and(missing_atoms, real_atoms)).item() == 0
