import torch

from tmol.system.pose import PackedBlockTypes, residue_types_from_residues
from tmol.score.hbond.HBondDependentTerm import HBondDependentTerm


def test_store_hbond_acc_don_designations_in_block_types(
    default_database, rts_ubq_res, torch_device
):

    rt_list = residue_types_from_residues(rts_ubq_res)

    hbdt = HBondDependentTerm.from_database(default_database, torch_device)
    for rt in rt_list:
        hbdt.setup_block_type(rt)

        assert hasattr(bt, "is_acceptor")
