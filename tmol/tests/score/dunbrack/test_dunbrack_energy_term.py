import numpy
import torch

from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    assert dunbrack_energy.device == torch_device


def test_annotate_block_types(ubq_res, default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    for bt in bt_list:
        dunbrack_energy.setup_block_type(bt)
    dunbrack_energy.setup_packed_block_types(pbt)

    assert hasattr(pbt, "dunbrack_packed_block_data")

    first_tensor = pbt.dunbrack_packed_block_data[0]

    assert first_tensor.device == torch_device
    dunbrack_energy.setup_packed_block_types(pbt)
    assert first_tensor is pbt.dunbrack_packed_block_data[0]


class TestDunbrackEnergyTerm(EnergyTermTestBase):
    energy_term_class = DunbrackEnergyTerm

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
            rts_ubq_res,
            default_database,
            torch_device,
            eps=1e-2,
            atol=4e-2,
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
            rts_ubq_res[0:4],
            default_database,
            torch_device,
            eps=1e-3,
            atol=2e-3,
            nondet_tol=1e-6,
        )
