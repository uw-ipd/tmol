import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.hbond import HBondScore, HBondParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_hbond_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(ubq_system, {HBondScore}, weights={"hbond": 1.0})


def test_hbond_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.hbond)

    src = ScoreSystem._build_with_modules(ubq_system, {HBondParameters})
    assert (
        HBondParameters.get(src).hbond_database
        is ParameterDatabase.get_default().scoring.hbond
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {HBondParameters}, hbond_database=clone_db
    )
    assert HBondParameters.get(src).hbond_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {HBondParameters})
    assert (
        HBondParameters.get(clone).hbond_database
        is HBondParameters.get(src).hbond_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {HBondParameters},
        hbond_database=ParameterDatabase.get_default().scoring.hbond,
    )
    assert (
        HBondParameters.get(clone).hbond_database
        is not HBondParameters.get(src).hbond_database
    )
    assert (
        HBondParameters.get(clone).hbond_database
        is ParameterDatabase.get_default().scoring.hbond
    )


def test_hbond_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {HBondScore}, weights={"hbond": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
