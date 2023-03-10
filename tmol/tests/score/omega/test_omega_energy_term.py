import numpy
import torch

from tmol.score.omega.omega_energy_term import OmegaEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):

    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)

    assert omega_energy.device == torch_device
    assert omega_energy.hb_param_db.global_param_table.device == torch_device
    assert omega_energy.hb_param_db.pair_param_table.device == torch_device
    assert omega_energy.hb_param_db.pair_poly_table.device == torch_device


def test_whole_pose_scoring_module_gradcheck_whole_pose(
    rts_ubq_res, default_database, torch_device
):

    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        omega_energy.setup_block_type(bt)
    omega_energy.setup_packed_block_types(p1.packed_block_types)
    omega_energy.setup_poses(p1)

    omega_pose_scorer = omega_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = omega_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-2, rtol=5e-3)


def test_whole_pose_scoring_module_10(rts_ubq_res, default_database, torch_device):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[6.741275]], dtype=numpy.float32), (n_poses))
    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        omega_energy.setup_block_type(bt)
    omega_energy.setup_packed_block_types(pn.packed_block_types)
    omega_energy.setup_poses(pn)

    omega_pose_scorer = omega_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = omega_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
