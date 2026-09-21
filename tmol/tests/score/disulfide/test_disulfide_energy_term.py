import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.disulfide import DisulfideEnergyTerm
from tmol.score.common import ZeroTermPoseScoringModule

from tmol.tests.score.common import EnergyTermTestBase


def test_smoke(default_database, torch_device: torch.device):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert disulfide_energy.device == torch_device
    assert disulfide_energy.global_params.a_mu.device == torch_device


def test_annotate_disulfide_conns(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

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


def test_pose_without_disulfides_uses_zero_term(
    ubq_pdb, default_database, torch_device: torch.device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)

    whole_pose = TestDisulfideEnergyTerm.get_whole_pose_scorer(
        pose_stack, default_database, torch_device
    )
    block_pair = TestDisulfideEnergyTerm.get_block_pair_scorer(
        pose_stack, default_database, torch_device
    )

    assert isinstance(whole_pose, ZeroTermPoseScoringModule)
    assert isinstance(block_pair, ZeroTermPoseScoringModule)
    torch.testing.assert_close(
        whole_pose(pose_stack.coords),
        torch.zeros((1, 1), device=torch_device),
    )
    torch.testing.assert_close(
        block_pair(pose_stack.coords),
        torch.zeros((1, 1, 10, 10), device=torch_device),
    )

    # Coordinate-independent terms still support the single-term benchmark
    # convention of calling backward() on their scalar output.
    coords = pose_stack.coords.detach().clone().requires_grad_(True)
    whole_pose(coords).sum().backward()


class TestDisulfideEnergyTerm(EnergyTermTestBase):
    energy_term_class = DisulfideEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, disulfide_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            disulfide_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        disulfide_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            disulfide_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls, disulfide_pdb, default_database, torch_device
    ):
        return super().test_whole_pose_scoring_gradcheck(
            disulfide_pdb,
            default_database,
            torch_device,
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, disulfide_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            disulfide_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, disulfide_pdb, default_database, torch_device):
        resnums = [(2, 4), (21, 23)]
        return super().test_block_scoring(
            disulfide_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, disulfide_pdb, default_database, torch_device
    ):
        resnums = [(2, 4), (21, 23)]
        return super().test_block_scoring_reweighted_gradcheck(
            disulfide_pdb, default_database, torch_device, resnums=resnums
        )
