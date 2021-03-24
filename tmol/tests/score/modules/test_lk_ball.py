import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.lk_ball import lk_ballScore, lk_ballParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_lk_ball_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(
            ubq_system, {lk_ballScore}, weights={"lk_ball": 1.0}
        )

    # TODO fordas add test assertions


def test_lk_ball_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.lk_ball)

    src = ScoreSystem._build_with_modules(ubq_system, {lk_ballParameters})
    assert (
        lk_ballParameters.get(src).lk_ball_database
        is ParameterDatabase.get_default().scoring.lk_ball
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {lk_ballParameters}, lk_ball_database=clone_db
    )
    assert lk_ballParameters.get(src).lk_ball_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {lk_ballParameters})
    assert (
        lk_ballParameters.get(clone).lk_ball_database
        is lk_ballParameters.get(src).lk_ball_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {lk_ballParameters},
        lk_ball_database=ParameterDatabase.get_default().scoring.lk_ball,
    )
    assert (
        lk_ballParameters.get(clone).lk_ball_database
        is not lk_ballParameters.get(src).lk_ball_database
    )
    assert (
        lk_ballParameters.get(clone).lk_ball_database
        is ParameterDatabase.get_default().scoring.lk_ball
    )


def test_lk_ball_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(
        twoubq, {lk_ballScore}, weights={"lk_ball": 1.0}
    )

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
