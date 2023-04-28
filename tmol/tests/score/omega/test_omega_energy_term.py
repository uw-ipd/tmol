import numpy
import torch

from tmol.score.omega.omega_energy_term import OmegaEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)

    assert omega_energy.device == torch_device
    assert omega_energy.global_params.K.device == torch_device


def test_annotate_omega_uaids(ubq_res, default_database, torch_device: torch.device):
    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    for bt in bt_list:
        omega_energy.setup_block_type(bt)
        assert hasattr(bt, "omega_quad_uaids")
    omega_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "omega_quad_uaids")
    assert pbt.omega_quad_uaids.device == torch_device


def test_whole_pose_scoring_module_gradcheck(
    rts_ubq_res, default_database, torch_device: torch.device
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


def test_whole_pose_scoring_module_single(
    rts_ubq_res, default_database, torch_device: torch.device
):
    gold_vals = numpy.array([[6.741275]], dtype=numpy.float32)
    omega_energy = OmegaEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        omega_energy.setup_block_type(bt)
    omega_energy.setup_packed_block_types(p1.packed_block_types)
    omega_energy.setup_poses(p1)

    omega_pose_scorer = omega_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = omega_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_10(
    rts_ubq_res, default_database, torch_device: torch.device
):
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
