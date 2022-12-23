from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.score.hbond.hbond_dependent_term import HBondDependentTerm


def test_hbond_dep_term_annotate_block_types_smoke(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)
    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)
        print("bt.hbbt_params", bt.name)
        print(bt.hbbt_params)


def test_hbond_dep_term_annotate_packed_block_types_smoke(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)
    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)
    hbdt.setup_packed_block_types(pbt)


def test_hbond_dep_term_setup_packed_block_types(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)

    hbdt.setup_packed_block_types(pbt)
