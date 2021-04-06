import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.cartbonded import CartBondedScore, CartBondedParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_cartbonded_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(
            ubq_system, {CartBondedScore}, weights={"cartbonded": 1.0}
        )


def test_cartbonded_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.cartbonded)

    src = ScoreSystem._build_with_modules(ubq_system, {CartBondedParameters})
    assert (
        CartBondedParameters.get(src).cartbonded_database
        is ParameterDatabase.get_default().scoring.cartbonded
    )

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {CartBondedParameters}, cartbonded_database=clone_db
    )
    assert CartBondedParameters.get(src).cartbonded_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {CartBondedParameters})
    assert (
        CartBondedParameters.get(clone).cartbonded_database
        is CartBondedParameters.get(src).cartbonded_database
    )

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {CartBondedParameters},
        cartbonded_database=ParameterDatabase.get_default().scoring.cartbonded,
    )
    assert (
        CartBondedParameters.get(clone).cartbonded_database
        is not CartBondedParameters.get(src).cartbonded_database
    )
    assert (
        CartBondedParameters.get(clone).cartbonded_database
        is ParameterDatabase.get_default().scoring.cartbonded
    )


def test_cartbonded_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(
        twoubq, {CartBondedScore}, weights={"cartbonded": 1.0}
    )

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2, 5)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
