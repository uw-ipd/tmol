import numpy
import torch

from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    assert elec_energy.device == torch_device
    assert elec_energy.param_resolver.device == torch_device


def test_annotate_restypes(ubq_res, default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, rt_list, torch_device
    )

    for rt in rt_list:
        elec_energy.setup_block_type(rt)
        assert hasattr(rt, "elec_partial_charge")
        assert hasattr(rt, "elec_inter_repr_path_distance")
        assert hasattr(rt, "elec_intra_repr_path_distance")
    elec_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "elec_partial_charge")
    assert hasattr(pbt, "elec_inter_repr_path_distance")
    assert hasattr(pbt, "elec_intra_repr_path_distance")

    assert pbt.elec_partial_charge.device == torch_device
    assert pbt.elec_inter_repr_path_distance.device == torch_device
    assert pbt.elec_intra_repr_path_distance.device == torch_device


def test_whole_pose_scoring_module_smoke(rts_ubq_res, default_database, torch_device):
    gold_vals = numpy.array([[-0.428092]], dtype=numpy.float32)
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res[0:3], device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        elec_energy.setup_block_type(bt)
    elec_energy.setup_packed_block_types(p1.packed_block_types)
    elec_energy.setup_poses(p1)

    elec_pose_scorer = elec_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = elec_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_gradcheck(
    rts_ubq_res, default_database, torch_device
):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res[0:4], device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        elec_energy.setup_block_type(bt)
    elec_energy.setup_packed_block_types(p1.packed_block_types)
    elec_energy.setup_poses(p1)

    elec_pose_scorer = elec_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = elec_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=5e-3, rtol=5e-3)


def test_whole_pose_scoring_module_10(rts_ubq_res, default_database, torch_device):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-135.45822]], dtype=numpy.float32), (n_poses))
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        elec_energy.setup_block_type(bt)
    elec_energy.setup_packed_block_types(pn.packed_block_types)
    elec_energy.setup_poses(pn)

    elec_pose_scorer = elec_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = elec_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
