import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert cartbonded_energy.device == torch_device


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

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
            ubq_pdb, default_database, torch_device, resnums=resnums
        )
