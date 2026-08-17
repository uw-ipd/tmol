import pytest
import torch

from tmol.tests import zero_padded_counts

from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score import beta2016_score_function
from tmol.score import ScoreFunction
from tmol.score.na_torsion import NaTorsionEnergyTerm


def _sfxn(variant, default_database, device):
    if variant == "na_torsion":
        sfxn = ScoreFunction(default_database, device)
        for st in NaTorsionEnergyTerm.score_types():
            sfxn.set_weight(st, 1.0)
        return sfxn

    beta2016 = beta2016_score_function(device, default_database)
    if variant == "beta2016":
        return beta2016

    # rebuild beta2016 without the DNA score types, so the term is never
    # created; zeroing its weight would leave it running and timed
    sfxn = ScoreFunction(default_database, device)
    dna = set(NaTorsionEnergyTerm.score_types())
    for st in beta2016.all_score_types():
        weight = float(beta2016.get_weight(st))
        if st not in dna and weight != 0:
            sfxn.set_weight(st, weight)
    return sfxn


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.parametrize("benchmark_pass", ["forward", "full"])
@pytest.mark.parametrize("variant", ["na_torsion", "beta2016_no_na", "beta2016"])
@pytest.mark.benchmark(group="na_torsion_score")
def test_na_torsion_benchmark(
    benchmark,
    benchmark_pass,
    variant,
    n_poses,
    dna_pdb,
    default_database,
    torch_device,
):
    n_poses = int(n_poses)
    pose_stack1 = pose_stack_from_pdb(dna_pdb, torch_device)
    pose_stack_n = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)

    sfxn = _sfxn(variant, default_database, torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)

    if benchmark_pass == "forward":

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.cpu()
            return scores

    elif benchmark_pass == "full":
        pose_stack_n.coords.requires_grad_(True)

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.backward(retain_graph=True)
            return scores.cpu()

    else:
        raise NotImplementedError

    score_pass
