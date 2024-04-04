import numpy
import torch

from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert disulfide_energy.device == torch_device
    assert disulfide_energy.global_params.a_mu.device == torch_device


def test_annotate_disulfide_conns(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(rts_disulfide_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    for bt in bt_list:
        disulfide_energy.setup_block_type(bt)
        assert hasattr(bt, "disulfide_connections")
    disulfide_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "disulfide_conns")
    disulfide_conns = pbt.disulfide_conns
    disulfide_energy.setup_packed_block_types(pbt)

    assert pbt.disulfide_conns.device == torch_device
    assert (
        pbt.disulfide_conns is disulfide_conns
    )  # Test to make sure the parameters remain the same instance


class TestDisulfideEnergyTerm(EnergyTermTestBase):
    energy_term_class = DisulfideEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(
        cls, rts_disulfide_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_whole_pose_scoring_10(
            rts_disulfide_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        rts_disulfide_res,
        default_database,
        torch_device: torch.device,
        update_baseline=False,
    ):
        return super().test_whole_pose_scoring_jagged(
            rts_disulfide_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls, rts_disulfide_res, default_database, torch_device
    ):
        return super().test_whole_pose_scoring_gradcheck(
            rts_disulfide_res,
            default_database,
            torch_device,
        )

    @classmethod
    def test_block_scoring(
        cls, rts_disulfide_res, default_database, torch_device, update_baseline=False
    ):
        rts_res = rts_disulfide_res[2:4] + rts_disulfide_res[21:23]
        return super().test_block_scoring(
            rts_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, rts_disulfide_res, default_database, torch_device
    ):
        rts_res = rts_disulfide_res[2:4] + rts_disulfide_res[21:23]
        return super().test_block_scoring_reweighted_gradcheck(
            rts_res,
            default_database,
            torch_device,
        )
