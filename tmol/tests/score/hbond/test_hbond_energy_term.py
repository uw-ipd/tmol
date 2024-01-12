import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    assert hbond_energy.device == torch_device
    assert hbond_energy.hb_param_db.global_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_poly_table.device == torch_device


def test_hbond_in_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    assert len(sfxn.all_terms()) == 1
    assert isinstance(sfxn.all_terms()[0], HBondEnergyTerm)


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    for rt in pbt.active_block_types:
        hbond_energy.setup_block_type(rt)
        assert hasattr(rt, "hbbt_params")
    hbond_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "hbpbt_params")

    assert pbt.hbpbt_params.tile_n_donH.device == torch_device
    assert pbt.hbpbt_params.tile_n_acc.device == torch_device
    assert pbt.hbpbt_params.tile_donH_inds.device == torch_device
    assert pbt.hbpbt_params.tile_acc_inds.device == torch_device
    assert pbt.hbpbt_params.tile_donorH_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_hybridization.device == torch_device
    assert pbt.hbpbt_params.is_hydrogen.device == torch_device


def test_whole_pose_scoring_module_smoke(ubq_pdb, default_database, torch_device):
    gold_vals = numpy.array([[-55.6756]], dtype=numpy.float32)
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(p1.packed_block_types)
    hbond_energy.setup_poses(p1)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = hbond_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_gradcheck_whole_pose(
    ubq_pdb, default_database, torch_device
):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(p1.packed_block_types)
    hbond_energy.setup_poses(p1)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = hbond_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-1, rtol=1e-1)


def test_whole_pose_scoring_module_gradcheck_partial_pose(
    ubq_pdb, default_database, torch_device
):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

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
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(p1.packed_block_types)
    hbond_energy.setup_poses(p1)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = hbond_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-3, rtol=1e-3)


def test_whole_pose_scoring_module_10(ubq_pdb, default_database, torch_device):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-55.6756]], dtype=numpy.float32), (n_poses))
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(pn.packed_block_types)
    hbond_energy.setup_poses(pn)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = hbond_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_jagged(ubq_pdb, default_database, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=40)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=60)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    for bt in poses.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(poses.packed_block_types)
    hbond_energy.setup_poses(poses)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(poses)
    coords = torch.nn.Parameter(poses.coords.clone())
    scores = hbond_pose_scorer(coords)

    gold_scores = torch.tensor(
        [[-24.6182, -41.5365]], dtype=torch.float32, device=torch_device
    )
    assert torch.allclose(gold_scores, scores, rtol=1e-4, atol=1e-4)
