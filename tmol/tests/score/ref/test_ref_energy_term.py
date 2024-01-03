import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.ref.ref_energy_term import RefEnergyTerm
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    assert ref_energy.device == torch_device


def test_annotate_block_types(
    fresh_default_packed_block_types, default_database, torch_device: torch.device
):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    bt_list = pbt.active_block_types

    for bt in bt_list:
        ref_energy.setup_block_type(bt)
        assert hasattr(bt, "ref_weight")
    ref_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "ref_weights")
    ref_weights = pbt.ref_weights
    ref_energy.setup_packed_block_types(pbt)

    assert pbt.ref_weights.device == torch_device
    assert (
        pbt.ref_weights is ref_weights
    )  # Test to make sure the parameters remain the same instance


def test_whole_pose_scoring_module_gradcheck(
    ubq_pdb, default_database, torch_device: torch.device
):
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        ref_energy.setup_block_type(bt)
    ref_energy.setup_packed_block_types(p1.packed_block_types)
    ref_energy.setup_poses(p1)

    ref_pose_scorer = ref_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = ref_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-2, rtol=5e-3)


def test_whole_pose_scoring_module_single(
    ubq_pdb, default_database, torch_device: torch.device
):
    gold_vals = numpy.array([[-41.275]], dtype=numpy.float32)
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        ref_energy.setup_block_type(bt)
    ref_energy.setup_packed_block_types(p1.packed_block_types)
    ref_energy.setup_poses(p1)

    ref_pose_scorer = ref_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = ref_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_10(
    ubq_pdb, default_database, torch_device: torch.device
):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-41.275]], dtype=numpy.float32), (n_poses))
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        ref_energy.setup_block_type(bt)
    ref_energy.setup_packed_block_types(pn.packed_block_types)

    ref_energy.setup_poses(pn)

    ref_pose_scorer = ref_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = ref_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_jagged(
    ubq_pdb, default_database, torch_device: torch.device
):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=60)
    p3 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=40)
    gold_vals = numpy.array(
        [
            [-41.275, -39.42759, -32.83142],
        ],
        dtype=numpy.float32,
    )
    ref_energy = RefEnergyTerm(param_db=default_database, device=torch_device)
    pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        ref_energy.setup_block_type(bt)
    ref_energy.setup_packed_block_types(pn.packed_block_types)
    ref_energy.setup_poses(pn)

    ref_pose_scorer = ref_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = ref_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
