import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.ljlk import LJScore, LJLKParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 1.0})


def test_ljlk_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.ljlk)

    src = ScoreSystem._build_with_modules(ubq_system, {LJLKParameters})
    assert (
        LJLKParameters.get(src).ljlk_database
        is ParameterDatabase.get_default().scoring.ljlk
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {LJLKParameters}, ljlk_database=clone_db
    )
    assert LJLKParameters.get(src).ljlk_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {LJLKParameters})
    assert (
        LJLKParameters.get(clone).ljlk_database is LJLKParameters.get(src).ljlk_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {LJLKParameters},
        ljlk_database=ParameterDatabase.get_default().scoring.ljlk,
    )
    assert (
        LJLKParameters.get(clone).ljlk_database
        is not LJLKParameters.get(src).ljlk_database
    )
    assert (
        LJLKParameters.get(clone).ljlk_database
        is ParameterDatabase.get_default().scoring.ljlk
    )


def test_lj_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {LJScore}, weights={"lj": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_subscores(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
