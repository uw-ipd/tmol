import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.lk_ball import LKBallScore, LKBallParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.system.score_support import score_method_to_even_weights_dict


@pytest.mark.benchmark(group="score_setup")
def test_lk_ball_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_system():
        return ScoreSystem.build_for(
            ubq_system,
            {LKBallScore},
            weights=score_method_to_even_weights_dict(LKBallScore),
        )


def test_lk_ball_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.ljlk)

    src = ScoreSystem._build_with_modules(ubq_system, {LKBallParameters})
    assert (
        LKBallParameters.get(src).ljlk_database
        is ParameterDatabase.get_default().scoring.ljlk
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {LKBallParameters}, ljlk_database=clone_db
    )
    assert LKBallParameters.get(src).ljlk_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {LKBallParameters})
    assert (
        LKBallParameters.get(clone).ljlk_database
        is LKBallParameters.get(src).ljlk_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {LKBallParameters},
        ljlk_database=ParameterDatabase.get_default().scoring.ljlk,
    )
    assert (
        LKBallParameters.get(clone).ljlk_database
        is not LKBallParameters.get(src).ljlk_database
    )
    assert (
        LKBallParameters.get(clone).ljlk_database
        is ParameterDatabase.get_default().scoring.ljlk
    )


def test_lk_ball_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(
        twoubq, {LKBallScore}, weights=score_method_to_even_weights_dict(LKBallScore)
    )

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_close(tot[0], tot[1])

    forward = stacked_score.intra_forward(coords)
    assert len(forward) == 4
    for terms in forward.values():
        assert len(terms) == 2
        torch.testing.assert_close(terms[0], terms[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
