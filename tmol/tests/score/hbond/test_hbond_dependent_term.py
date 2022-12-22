from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.score.hbond.hbond_dependent_term import HBondDependentTerm


def test_hbond_dep_term_annotate_block_types_smoke(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)
    hbdt = HBondDependentTerm.from_database(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)


def test_store_hbond_acc_don_designations_in_block_types(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)

    hbdt = HBondDependentTerm.from_database(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)

        assert hasattr(bt, "hbbt_params")
        assert bt.hbbt_params.is_acceptor.shape == (bt.n_atoms,)
        assert bt.hbbt_params.is_donor.shape == (bt.n_atoms,)
        assert bt.hbbt_params.acceptor_type.shape == (bt.n_atoms,)
        assert bt.hbbt_params.donor_type.shape == (bt.n_atoms,)
        assert bt.hbbt_params.acceptor_hybridization.shape == (bt.n_atoms,)
        assert bt.hbbt_params.is_hydrogen.shape == (bt.n_atoms,)
        assert bt.hbbt_params.donor_attached_hydrogens.shape[0] == bt.n_atoms


def test_hbond_dep_term_setup_packed_block_types(
    default_database, rts_ubq_res, torch_device
):

    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    hbdt = HBondDependentTerm.from_database(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)

    hbdt.setup_packed_block_types(pbt)
