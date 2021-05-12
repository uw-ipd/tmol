import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.rama import RamaScore, RamaParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(ubq_system, {RamaScore}, weights={"lj": 1.0})


def test_rama_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.rama)

    src = ScoreSystem._build_with_modules(ubq_system, {RamaParameters})
    assert (
        RamaParameters.get(src).rama_database
        is ParameterDatabase.get_default().scoring.rama
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {RamaParameters}, rama_database=clone_db
    )
    assert RamaParameters.get(src).rama_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {RamaParameters})
    assert (
        RamaParameters.get(clone).rama_database is RamaParameters.get(src).rama_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {RamaParameters},
        rama_database=ParameterDatabase.get_default().scoring.rama,
    )
    assert (
        RamaParameters.get(clone).rama_database
        is not RamaParameters.get(src).rama_database
    )
    assert (
        RamaParameters.get(clone).rama_database
        is ParameterDatabase.get_default().scoring.rama
    )


def test_rama_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {RamaScore}, weights={"rama": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_subscores(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
