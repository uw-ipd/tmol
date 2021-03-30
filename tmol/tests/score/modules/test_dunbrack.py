import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.dunbrack import LJScore, dunbrackParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 1.0})

    # TODO fordas add test assertions


def test_dunbrack_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.dunbrack)

    src = ScoreSystem._build_with_modules(ubq_system, {dunbrackParameters})
    assert (
        dunbrackParameters.get(src).dunbrack_database
        is ParameterDatabase.get_default().scoring.dunbrack
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {dunbrackParameters}, dunbrack_database=clone_db
    )
    assert dunbrackParameters.get(src).dunbrack_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {dunbrackParameters})
    assert (
        dunbrackParameters.get(clone).dunbrack_database
        is dunbrackParameters.get(src).dunbrack_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {dunbrackParameters},
        dunbrack_database=ParameterDatabase.get_default().scoring.dunbrack,
    )
    assert (
        dunbrackParameters.get(clone).dunbrack_database
        is not dunbrackParameters.get(src).dunbrack_database
    )
    assert (
        dunbrackParameters.get(clone).dunbrack_database
        is ParameterDatabase.get_default().scoring.dunbrack
    )


def test_lj_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {LJScore}, weights={"lj": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
