import pytest
import torch

from tmol.tests.torch import zero_padded_counts

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score.dna_dihedral.dna_dihedral_energy_term import DnaDihedralEnergyTerm


def _sfxn(variant, default_database, device):
    from tmol.score import _non_memoized_beta2016

    if variant == "dna_torsion":
        sfxn = ScoreFunction(default_database, device)
        for st in DnaDihedralEnergyTerm.score_types():
            sfxn.set_weight(st, 1.0)
        return sfxn

    sfxn = _non_memoized_beta2016(device, default_database)
    if variant == "beta2016_no_dna":
        # a zero weight drops the term entirely, so this measures beta2016
        # as it was before dna_torsion existed
        sfxn.set_weight(ScoreType.dna_torsion, 0.0)
    return sfxn


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.parametrize("benchmark_pass", ["forward", "full"])
@pytest.mark.parametrize("variant", ["dna_torsion", "beta2016_no_dna", "beta2016"])
@pytest.mark.benchmark(group="dna_torsion_score")
def test_dna_torsion_benchmark(
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
