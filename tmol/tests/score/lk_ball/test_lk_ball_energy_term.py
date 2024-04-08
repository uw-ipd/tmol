import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device):
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)

    assert lk_ball_energy.device == torch_device


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types

    first_params = {}
    for bt in pbt.active_block_types:
        lk_ball_energy.setup_block_type(bt)
        assert hasattr(bt, "hbbt_params")
        first_params[bt.name] = bt.hbbt_params

    for bt in pbt.active_block_types:
        # test that block-type annotation is not repeated;
        # original annotation is still present in the bt
        lk_ball_energy.setup_block_type(bt)
        assert first_params[bt.name] is bt.hbbt_params

    lk_ball_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "lk_ball_params")

    init_pbt_lk_ball_params = pbt.lk_ball_params
    lk_ball_energy.setup_packed_block_types(pbt)
    # test that the initial packed-block-types annotation
    # has not been repeated; initial annotation is still
    # present in the pbt
    assert init_pbt_lk_ball_params is pbt.lk_ball_params

    assert pbt.lk_ball_params.tile_n_polar_atoms.device == torch_device
    assert pbt.lk_ball_params.tile_n_occluder_atoms.device == torch_device
    assert pbt.lk_ball_params.tile_pol_occ_inds.device == torch_device
    assert pbt.lk_ball_params.tile_lk_ball_params.device == torch_device


def test_whole_pose_scoring_module_smoke(ubq_pdb, default_database, torch_device):
    gold_vals = numpy.array(
        [[422.0388], [172.1965], [1.5786], [10.9946]], dtype=numpy.float32
    )
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        lk_ball_energy.setup_block_type(bt)
    lk_ball_energy.setup_packed_block_types(p1.packed_block_types)
    lk_ball_energy.setup_poses(p1)

    lk_ball_pose_scorer = lk_ball_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = lk_ball_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-3, rtol=1e-3
    )


class TestLKBallEnergyTerm(EnergyTermTestBase):
    energy_term_class = LKBallEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            nondet_tol=1e-6,  # fd this is necessary here...
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
    def test_block_scoring_matches_full_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_full_pose_scoring(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            update_baseline=False,
            resnums=resnums,
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
            nondet_tol=1e-6,  # fd this is necessary here...
        )
