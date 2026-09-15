import pytest
import torch

from tmol.score.ljlk import LJLKEnergyTerm
from tmol.score.elec import ElecEnergyTerm
from tmol.pose import PackedBlockTypes

from tmol.tests.score.common import EnergyTermTestBase, pose_stack_from_pdb_and_resnums


def test_smoke(default_database, torch_device):
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    assert ljlk_energy.type_params.lj_radius.device == torch_device
    assert ljlk_energy.global_params.max_dis.device == torch_device


def test_beta_nov16_water_ljlk_parameters(default_database):
    params = {
        row.name: row for row in default_database.scoring.ljlk.atom_type_parameters
    }
    assert (
        params["Owat"].lj_radius,
        params["Owat"].lj_wdepth,
        params["Owat"].lk_dgfree,
    ) == (1.542743, 0.161947, -4.5480)
    assert (params["Hwat"].lj_radius, params["Hwat"].lj_wdepth) == (
        0.901681,
        0.01,
    )


@pytest.mark.parametrize(
    "term_class,gold",
    [
        (LJLKEnergyTerm, [[-12.7234735], [1.9350461], [18.1323757]]),
        (ElecEnergyTerm, [[-4.9202342]]),
    ],
)
def test_water_box_nonbonded_score(
    water_box_pdb, default_database, torch_device, term_class, gold
):
    pose = pose_stack_from_pdb_and_resnums(water_box_pdb, torch_device, [(0, 31)])
    term = term_class(default_database, torch_device)
    for block_type in pose.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)

    score = term.render_whole_pose_scoring_module(pose)(pose.coords)

    torch.testing.assert_close(score, score.new_tensor(gold), rtol=2e-5, atol=2e-5)


def test_annotate_heavy_ats_in_tile(
    fresh_default_restype_set, default_database, torch_device
):
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    for rt in fresh_default_restype_set.residue_types:
        ljlk_energy.setup_block_type(rt)
        assert hasattr(rt, "ljlk_heavy_atoms_in_tile")
        assert hasattr(rt, "ljlk_n_heavy_atoms_in_tile")
    ljlk_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "ljlk_heavy_atoms_in_tile")
    assert hasattr(pbt, "ljlk_n_heavy_atoms_in_tile")


class TestLJLKEnergyTerm(EnergyTermTestBase):
    energy_term_class = LJLKEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums
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
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb, default_database, torch_device
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
            nondet_tol=1e-6,
        )
