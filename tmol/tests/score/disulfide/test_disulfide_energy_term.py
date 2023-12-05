import numpy
import torch

from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device: torch.device):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert disulfide_energy.device == torch_device
    assert disulfide_energy.global_params.a_mu.device == torch_device


def test_annotate_disulfide_conns(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(rts_disulfide_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, bt_list, torch_device
    )

    for bt in bt_list:
        disulfide_energy.setup_block_type(bt)
        assert hasattr(bt, "disulfide_connections")
    disulfide_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "disulfide_conns")
    disulfide_conns = pbt.disulfide_conns
    disulfide_energy.setup_packed_block_types(pbt)

    assert pbt.disulfide_conns.device == torch_device
    assert (
        pbt.disulfide_conns is disulfide_conns
    )  # Test to make sure the parameters remain the same instance


def test_whole_pose_scoring_module_gradcheck_whole_pose(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        disulfide_energy.setup_block_type(bt)
    disulfide_energy.setup_packed_block_types(p1.packed_block_types)
    disulfide_energy.setup_poses(p1)

    disulfide_pose_scorer = disulfide_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = disulfide_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-2, rtol=5e-3)


def test_whole_pose_scoring_module_single(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    gold_vals = numpy.array([[-3.25716]], dtype=numpy.float32)
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        disulfide_energy.setup_block_type(bt)
    disulfide_energy.setup_packed_block_types(p1.packed_block_types)
    disulfide_energy.setup_poses(p1)

    disulfide_pose_scorer = disulfide_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = disulfide_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_10(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-3.25716]], dtype=numpy.float32), (n_poses))
    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        disulfide_energy.setup_block_type(bt)
    disulfide_energy.setup_packed_block_types(pn.packed_block_types)

    disulfide_energy.setup_poses(pn)

    disulfide_pose_scorer = disulfide_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = disulfide_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_jagged(
    rts_disulfide_res, default_database, torch_device: torch.device
):
    rts_disulfide_60 = rts_disulfide_res[:60]
    rts_disulfide_33 = rts_disulfide_res[:33]
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_res, device=torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_60, device=torch_device
    )
    p3 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_disulfide_33, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

    disulfide_energy = DisulfideEnergyTerm(
        param_db=default_database, device=torch_device
    )
    for bt in pn.packed_block_types.active_block_types:
        disulfide_energy.setup_block_type(bt)
    disulfide_energy.setup_packed_block_types(pn.packed_block_types)
    disulfide_energy.setup_poses(pn)

    disulfide_pose_scorer = disulfide_energy.render_whole_pose_scoring_module(pn)

    disulfide_pose_scorer(pn.coords)

    print("ok!")
