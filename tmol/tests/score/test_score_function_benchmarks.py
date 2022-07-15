import pytest
import torch
import numpy

from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack import PoseStack

from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm


@pytest.mark.parametrize("energy_term", [LJLKEnergyTerm], ids=["ljlk"])
@pytest.mark.parametrize("n_poses", [1, 3, 10, 30, 100])
@pytest.mark.benchmark(group="setup_res_centric_scoring")
def test_res_centric_score_benchmark_setup(
    benchmark, energy_term, n_poses, rts_ubq_res, default_database, torch_device
):
    pose_stack1 = PoseStack.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )

    pose_stack_n = PoseStack.from_poses([pose_stack1] * n_poses, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)

    for st in energy_term.score_types():
        sfxn.set_weight(st, 1.0)

    @benchmark
    def render_whole_pose_scoring_module():
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)


@pytest.mark.parametrize("energy_term", [LJLKEnergyTerm], ids=["ljlk"])
@pytest.mark.parametrize("benchmark_pass", ["forward"])
@pytest.mark.parametrize("n_poses", [10, 30, 100])
@pytest.mark.benchmark(group="res_centric_score_components")
def dont_test_res_centric_score_benchmark(
    benchmark,
    benchmark_pass,
    energy_term,
    n_poses,
    rts_ubq_res,
    default_database,
    torch_device,
):
    pose_stack1 = PoseStack.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )
    pose_stack_n = PoseStack.from_poses([pose_stack1] * n_poses, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)

    for st in energy_term.score_types():
        sfxn.set_weight(st, 1.0)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)

    if benchmark_pass == "full":

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.backward(retain_graph=True)
            return scores

    elif benchmark_pass == "forward":

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            return scores

    elif benchmark_pass == "backward":
        scores = torch.sum(scorer(pose_stack_n.coords))

        @benchmark
        def score_pass():
            scores.backward(retain_graph=True)
            return scores

    else:
        raise NotImplementedError

    # print("scores")
    # print(scores[:,:3])
    # print(scores.shape)
