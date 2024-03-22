import numpy
import torch
from torch._C import device

from tmol.score.ref.ref_energy_term import RefEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    assert ref_energy.device == torch_device


def test_annotate_block_types(
    rts_ubq_res, default_database, torch_device: torch.device
):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    for bt in bt_list:
        ref_energy.setup_block_type(bt)
        assert hasattr(bt, "ref_weight")
    ref_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "ref_weights")
    ref_weights = pbt.ref_weights
    ref_energy.setup_packed_block_types(pbt)

    assert pbt.ref_weights.device == torch_device
    assert (
        pbt.ref_weights is ref_weights
    )  # Test to make sure the parameters remain the same instance


class TestRefEnergyTerm(EnergyTermTestBase):
    energy_term_class = RefEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_whole_pose_scoring_10(
            rts_ubq_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        rts_ubq_res,
        default_database,
        torch_device: torch.device,
        update_baseline=False,
    ):
        return super().test_whole_pose_scoring_jagged(
            rts_ubq_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_whole_pose_scoring_gradcheck(
            rts_ubq_res, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_block_scoring(
            rts_ubq_res[0:4], default_database, torch_device, update_baseline
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_block_scoring_reweighted_gradcheck(
            rts_ubq_res[0:4], default_database, torch_device
        )
