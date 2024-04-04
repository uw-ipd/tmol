import numpy
import torch

from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert cartbonded_energy.device == torch_device


def test_annotate_restypes(ubq_res, default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    for bt in bt_list:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pbt)

    assert hasattr(pbt, "cartbonded_subgraphs")
    assert hasattr(pbt, "cartbonded_subgraph_offsets")
    assert hasattr(pbt, "cartbonded_max_subgraphs_per_block")
    assert hasattr(pbt, "cartbonded_params_hash_keys")
    assert hasattr(pbt, "cartbonded_params_hash_values")

    assert pbt.cartbonded_subgraphs.device == torch_device
    assert pbt.cartbonded_subgraph_offsets.device == torch_device
    assert pbt.cartbonded_params_hash_keys.device == torch_device
    assert pbt.cartbonded_params_hash_values.device == torch_device

    cartbonded_subgraphs = pbt.cartbonded_subgraphs
    cartbonded_energy.setup_packed_block_types(pbt)
    assert cartbonded_subgraphs is pbt.cartbonded_subgraphs


class TestCartBondedEnergyTerm(EnergyTermTestBase):
    energy_term_class = CartBondedEnergyTerm

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
        rts_ubq_res = rts_ubq_res[0:2]
        return super().test_whole_pose_scoring_gradcheck(
            rts_ubq_res,
            default_database,
            torch_device,
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
        )
