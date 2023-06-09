import numpy
import torch

from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert cartbonded_energy.device == torch_device


def test_annotate_cartbonded_uaids(
    ubq_res, default_database, torch_device: torch.device
):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    for bt in bt_list:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pbt)
    """
    assert hasattr(pbt, "cartbonded_quad_uaids")
    assert pbt.cartbonded_quad_uaids.device == torch_device"""


def test_whole_pose_scoring_module_gradcheck(
    rts_ubq_res, default_database, torch_device: torch.device
):
    rts_ubq_res = rts_ubq_res[0:2]
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(p1.packed_block_types)
    cartbonded_energy.setup_poses(p1)

    cartbonded_pose_scorer = cartbonded_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = cartbonded_pose_scorer(coords)[2:3]
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-2, atol=5e-2)


def test_whole_pose_scoring_module_single(
    rts_ubq_res, default_database, torch_device: torch.device
):
    # rts_ubq_res = rts_ubq_res[0:2]
    gold_vals = numpy.array([[0.0], [0.0], [0.0], [0.0]], dtype=numpy.float32)
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(p1.packed_block_types)
    cartbonded_energy.setup_poses(p1)

    cartbonded_pose_scorer = cartbonded_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = cartbonded_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_10(
    rts_ubq_res, default_database, torch_device: torch.device
):
    n_poses = 10
    gold_vals = numpy.tile(
        numpy.array([[0.0], [0.0], [0.0]], dtype=numpy.float32), (n_poses)
    )
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pn.packed_block_types)
    cartbonded_energy.setup_poses(pn)

    cartbonded_pose_scorer = cartbonded_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = cartbonded_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
