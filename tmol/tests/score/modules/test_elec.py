import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.elec import ElecScore, ElecParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_elec_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_system():
        return ScoreSystem.build_for(ubq_system, {ElecScore}, weights={"elec": 1.0})


def test_elec_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.elec)

    src = ScoreSystem._build_with_modules(ubq_system, {ElecParameters})
    assert (
        ElecParameters.get(src).elec_database
        is ParameterDatabase.get_default().scoring.elec
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {ElecParameters}, elec_database=clone_db
    )
    assert ElecParameters.get(src).elec_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {ElecParameters})
    assert (
        ElecParameters.get(clone).elec_database is ElecParameters.get(src).elec_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {ElecParameters},
        elec_database=ParameterDatabase.get_default().scoring.elec,
    )
    assert (
        ElecParameters.get(clone).elec_database
        is not ElecParameters.get(src).elec_database
    )
    assert (
        ElecParameters.get(clone).elec_database
        is ParameterDatabase.get_default().scoring.elec
    )


def test_elec_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {ElecScore}, weights={"elec": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_close(tot[0], tot[1])

    forward = stacked_score.intra_forward(coords)
    assert len(forward) == 1
    for terms in forward.values():
        assert len(terms) == 2
        torch.testing.assert_close(terms[0], terms[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
