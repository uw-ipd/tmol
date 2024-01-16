import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm

from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


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


def test_whole_pose_scoring_module_gradcheck_partial_pose(
    ubq_pdb, default_database, torch_device
):
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)
    res_not_term = torch.zeros((1, 6, 2), dtype=torch.bool, device=torch_device)
    res_not_term[0, 0, 0] = True
    res_not_term[0, 5, 1] = True
    p1 = pose_stack_from_pdb(
        ubq_pdb,
        torch_device,
        residue_start=6,
        residue_end=12,
        res_not_connected=res_not_term,
    )
    for bt in p1.packed_block_types.active_block_types:
        lk_ball_energy.setup_block_type(bt)
    lk_ball_energy.setup_packed_block_types(p1.packed_block_types)
    lk_ball_energy.setup_poses(p1)

    lk_ball_pose_scorer = lk_ball_energy.render_whole_pose_scoring_module(p1)

    weights = torch.tensor(
        [[0.75], [1.25], [0.625], [0.8125]], dtype=torch.float32, device=torch_device
    )

    def score(coords):
        scores = lk_ball_pose_scorer(coords)

        wtd_score = torch.sum(weights * scores)
        return wtd_score

    gradcheck(
        score,
        (p1.coords.requires_grad_(True),),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
        nondet_tol=1e-3,
    )


def test_whole_pose_scoring_module_10(ubq_pdb, default_database, torch_device):
    n_poses = 10
    gold_vals = numpy.tile(
        numpy.array(
            [[422.0388], [172.1965], [1.5786], [10.9946]],
            dtype=numpy.float32,
        ),
        (n_poses),
    )
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        lk_ball_energy.setup_block_type(bt)
    lk_ball_energy.setup_packed_block_types(pn.packed_block_types)
    lk_ball_energy.setup_poses(pn)

    lk_ball_pose_scorer = lk_ball_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = lk_ball_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_jagged(ubq_pdb, default_database, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=40)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=60)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    lk_ball_energy = LKBallEnergyTerm(param_db=default_database, device=torch_device)
    for bt in poses.packed_block_types.active_block_types:
        lk_ball_energy.setup_block_type(bt)
    lk_ball_energy.setup_packed_block_types(poses.packed_block_types)
    lk_ball_energy.setup_poses(poses)

    lk_ball_pose_scorer = lk_ball_energy.render_whole_pose_scoring_module(poses)
    coords = torch.nn.Parameter(poses.coords.clone())
    scores = lk_ball_pose_scorer(coords)

    gold_scores = torch.tensor(
        [[196.0713, 323.4101], [80.7462, 129.9590], [0.4837, 0.9545], [3.3417, 6.2497]],
        dtype=torch.float32,
        device=torch_device,
    )
    assert torch.allclose(gold_scores, scores, rtol=1e-4, atol=1e-4)
