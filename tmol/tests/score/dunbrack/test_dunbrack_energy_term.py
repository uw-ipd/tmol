import numpy
import torch

from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

# from tmol.tests.autograd import GradcheckError
from tmol.tests.autograd import gradcheck

import ast


def test_smoke(default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    assert dunbrack_energy.device == torch_device
    assert dunbrack_energy.global_params.K.device == torch_device


def test_annotate_block_types(ubq_res, default_database, torch_device: torch.device):
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)

    bt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    for bt in bt_list:
        dunbrack_energy.setup_block_type(bt)
        # assert hasattr(bt, "dunbrack_quad_uaids")
    dunbrack_energy.setup_packed_block_types(pbt)
    # assert hasattr(pbt, "dunbrack_quad_uaids")
    # assert pbt.dunbrack_quad_uaids.device == torch_device


def test_whole_pose_scoring_module_gradcheck(
    rts_ubq_res, default_database, torch_device: torch.device
):
    rts_ubq_res = rts_ubq_res[3:5]
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        dunbrack_energy.setup_block_type(bt)
    dunbrack_energy.setup_packed_block_types(p1.packed_block_types)
    dunbrack_energy.setup_poses(p1)

    dunbrack_pose_scorer = dunbrack_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = dunbrack_pose_scorer(coords)
        return torch.sum(scores)

    try:
        gradcheck(
            score,
            (p1.coords.requires_grad_(True),),
            eps=1e-2,
            atol=1e-2,
            raise_exception=True,
        )
        pass
    except RuntimeError as e:
        err_str = e.args[0]
        numerical, analytical = err_str.split("analytical:tensor")
        numerical = numerical.split("numerical:tensor")[-1]

        numerical = ast.literal_eval(numerical)
        analytical = ast.literal_eval(analytical)

        maxdif = 0
        for ind, num, ana in zip(range(len(numerical)), numerical, analytical):
            num = num[0]
            ana = ana[0]
            dif = abs(num - ana)
            maxdif = max(dif, maxdif)
            # print("{:3d}  {: .5f}  {: .5f}  {: .5f}".format(ind, num, ana, dif))
        print("Maximum difference: %f" % (maxdif))


def test_whole_pose_scoring_module_single(
    rts_ubq_res, default_database, torch_device: torch.device
):
    gold_vals = numpy.array([[70.6497], [240.3100], [99.6609]], dtype=numpy.float32)
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        dunbrack_energy.setup_block_type(bt)
    dunbrack_energy.setup_packed_block_types(p1.packed_block_types)
    dunbrack_energy.setup_poses(p1)

    dunbrack_pose_scorer = dunbrack_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = dunbrack_pose_scorer(coords)

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
        numpy.array([[70.6497], [240.3100], [99.6609]], dtype=numpy.float32), (n_poses)
    )
    dunbrack_energy = DunbrackEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        dunbrack_energy.setup_block_type(bt)
    dunbrack_energy.setup_packed_block_types(pn.packed_block_types)
    dunbrack_energy.setup_poses(pn)

    dunbrack_pose_scorer = dunbrack_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = dunbrack_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )
