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


def test_annotate_restypes(ubq_res, default_database, torch_device: torch.device):
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    for bt in bt_list:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pbt)

    assert hasattr(pbt, "cartbonded_subgraphs")
    assert hasattr(pbt, "cartbonded_subgraph_offsets")
    assert hasattr(pbt, "cartbonded_max_subgraphs_per_block")
    assert hasattr(pbt, "cartbonded_params_hash_keys")
    assert hasattr(pbt, "cartbonded_params_hash_values")

    assert pbt.cartbonded_subgraphs.device == torch_device
    assert pbt.cartbonded_subgraph_offsets.device == torch_device
    assert pbt.cartbonded_params_hash_keys.device == torch_device
    assert pbt.cartbonded_params_hash_values.device == torch_device

    cartbonded_subgraphs = pbt.cartbonded_subgraphs
    cartbonded_energy.setup_packed_block_types(pbt)
    assert cartbonded_subgraphs is pbt.cartbonded_subgraphs


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
    gold_vals = numpy.array(
        [[37.7848], [183.5777], [50.5842], [9.4305], [47.4197]], dtype=numpy.float32
    )
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
        numpy.array(
            [[37.7848], [183.5777], [50.5842], [9.4305], [47.4197]], dtype=numpy.float32
        ),
        (n_poses),
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


def test_whole_pose_scoring_module_jagged(
    rts_ubq_res, default_database, torch_device: torch.device
):
    rts_ubq_60 = rts_ubq_res[:60]
    rts_ubq_40 = rts_ubq_res[:40]
    gold_vals = numpy.array(
        [
            [37.7848, 30.0712, 19.7317],
            [183.5777, 149.7569, 107.8989],
            [50.5842, 38.3034, 24.6760],
            [9.4305, 6.9274, 5.4583],
            [47.4197, 38.3253, 29.3032],
        ],
        dtype=numpy.float32,
    )
    cartbonded_energy = CartBondedEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_60, device=torch_device
    )
    p3 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_40, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        cartbonded_energy.setup_block_type(bt)
    cartbonded_energy.setup_packed_block_types(pn.packed_block_types)
    cartbonded_energy.setup_poses(pn)

    cartbonded_pose_scorer = cartbonded_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = cartbonded_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    print(scores.cpu().detach().numpy())

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
