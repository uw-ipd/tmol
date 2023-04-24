import numpy
import torch

from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device):

    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    assert hbond_energy.device == torch_device
    assert hbond_energy.hb_param_db.global_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_poly_table.device == torch_device


def test_annotate_restypes(ubq_res, default_database, torch_device):

    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

    for rt in rt_list:
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


def test_whole_pose_scoring_module_smoke(rts_ubq_res, default_database, torch_device):
    gold_vals = numpy.array([[-54.8584]], dtype=numpy.float32)
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
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
    rts_ubq_res, default_database, torch_device
):

    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
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
    rts_ubq_res, default_database, torch_device
):

    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res[6:12], device=torch_device
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


def test_whole_pose_scoring_module_10(rts_ubq_res, default_database, torch_device):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-54.8584]], dtype=numpy.float32), (n_poses))
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
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
