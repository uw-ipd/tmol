import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    assert dunbrack_energy.device == torch_device


def test_annotate_block_types(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

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
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        ubq_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )
