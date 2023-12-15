import numpy

from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.score.hbond.hbond_dependent_term import HBondDependentTerm


def test_hbond_dep_term_annotate_block_types_smoke(
    default_database, rts_ubq_res, torch_device
):
    bt_list = residue_types_from_residues(rts_ubq_res)
    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)
        assert hasattr(bt, "hbbt_params")


def test_hbond_dep_term_annotate_packed_block_types_smoke(
    default_database, rts_ubq_res, torch_device
):
    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )
    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)
    hbdt.setup_packed_block_types(pbt)


def test_hbond_dep_term_setup_packed_block_types(
    default_database, rts_ubq_res, torch_device
):
    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    hbdt = HBondDependentTerm(default_database, torch_device)
    for bt in bt_list:
        hbdt.setup_block_type(bt)

    hbdt.setup_packed_block_types(pbt)


def test_hbond_dep_term_setup_ser_block_type(
    default_database, rts_ubq_res, torch_device
):
    bt_list = residue_types_from_residues(rts_ubq_res)
    ser_bt = next(bt for bt in bt_list if bt.name == "SER")

    hbdt = HBondDependentTerm(default_database, torch_device)
    hbdt.setup_block_type(ser_bt)

    assert hasattr(ser_bt, "hbbt_params")
    ser_hbbt_params = ser_bt.hbbt_params

    numpy.testing.assert_equal(
        numpy.array([2], dtype=numpy.int32), ser_hbbt_params.tile_n_donH
    )
    numpy.testing.assert_equal(
        numpy.array([2], dtype=numpy.int32), ser_hbbt_params.tile_n_don_hvy
    )
    numpy.testing.assert_equal(
        numpy.array([2], dtype=numpy.int32), ser_hbbt_params.tile_n_acc
    )

    def at_ind(target):
        return ser_bt.atom_to_idx[target]

    gold_H_inds = numpy.full((1, 32), -1, dtype=numpy.int32)
    gold_H_inds[0, 0] = at_ind("H")
    gold_H_inds[0, 1] = at_ind("HG")
    numpy.testing.assert_equal(gold_H_inds, ser_hbbt_params.tile_donH_inds)

    gold_hvy_don_inds = numpy.full((1, 32), -1, dtype=numpy.int32)
    gold_hvy_don_inds[0, 0] = at_ind("N")
    gold_hvy_don_inds[0, 1] = at_ind("OG")
    numpy.testing.assert_equal(gold_hvy_don_inds, ser_hbbt_params.tile_don_hvy_inds)
    numpy.testing.assert_equal(gold_hvy_don_inds, ser_hbbt_params.tile_donH_hvy_inds)

    gold_tile_which_donH_of_donH_hvy = numpy.full((1, 32), -1, dtype=numpy.int32)
    gold_tile_which_donH_of_donH_hvy[0, 0] = 0
    gold_tile_which_donH_of_donH_hvy[0, 1] = 0
    numpy.testing.assert_equal(
        gold_tile_which_donH_of_donH_hvy, ser_hbbt_params.tile_which_donH_of_donH_hvy
    )

    gold_acc_inds = numpy.full((1, 32), -1, dtype=numpy.int32)
    gold_acc_inds[0, 0] = at_ind("O")
    gold_acc_inds[0, 1] = at_ind("OG")
    numpy.testing.assert_equal(gold_acc_inds, ser_hbbt_params.tile_acc_inds)

    gold_tile_acceptor_n_attached_H = numpy.full((1, 32), -1, dtype=numpy.int32)
    gold_tile_acceptor_n_attached_H[0, 0] = 0
    gold_tile_acceptor_n_attached_H[0, 1] = 1
    numpy.testing.assert_equal(
        gold_tile_acceptor_n_attached_H, ser_hbbt_params.tile_acceptor_n_attached_H
    )
