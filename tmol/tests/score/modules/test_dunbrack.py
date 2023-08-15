import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.dunbrack import DunbrackScore, DunbrackParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_dunbrack_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_system():
        return ScoreSystem.build_for(
            ubq_system,
            {DunbrackScore},
            weights={
                "dunbrack_rot": 1.0,
                "dunbrack_rotdev": 2.0,
                "dunbrack_semirot": 3.0,
            },
        )


def test_dunbrack_rotamer_library_clone_factory(ubq_system):
    clone_dunbrack_rotamer_library = copy.copy(
        ParameterDatabase.get_default().scoring.dun
    )

    src = ScoreSystem._build_with_modules(ubq_system, {DunbrackParameters})
    assert (
        DunbrackParameters.get(src).dunbrack_rotamer_library
        is ParameterDatabase.get_default().scoring.dun
    )

    # Rotamer library is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system,
        {DunbrackParameters},
        dunbrack_rotamer_library=clone_dunbrack_rotamer_library,
    )
    assert (
        DunbrackParameters.get(src).dunbrack_rotamer_library
        is clone_dunbrack_rotamer_library
    )

    # Rotamer library is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {DunbrackParameters})
    assert (
        DunbrackParameters.get(clone).dunbrack_rotamer_library
        is DunbrackParameters.get(src).dunbrack_rotamer_library
    )

    # Rotamer library is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src,
        {DunbrackParameters},
        dunbrack_rotamer_library=ParameterDatabase.get_default().scoring.dun,
    )
    assert (
        DunbrackParameters.get(clone).dunbrack_rotamer_library
        is not DunbrackParameters.get(src).dunbrack_rotamer_library
    )
    assert (
        DunbrackParameters.get(clone).dunbrack_rotamer_library
        is ParameterDatabase.get_default().scoring.dun
    )


def test_dunbrack_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(
        twoubq,
        {DunbrackScore},
        weights={"dunbrack_rot": 1.0, "dunbrack_rotdev": 2.0, "dunbrack_semirot": 3.0},
    )

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    forward = stacked_score.intra_forward(coords)
    assert len(forward) == 3
    for terms in forward.values():
        assert len(terms) == 2
        torch.testing.assert_allclose(terms[0], terms[1])

    sumtot = torch.sum(tot)
    sumtot.backward()


def test_dunbrack_for_jagged_system(ubq_res):
    ubq_system1 = PackedResidueSystem.from_residues(ubq_res[0:40])
    ubq_system2 = PackedResidueSystem.from_residues(ubq_res[0:60])

    twoubq = PackedResidueSystemStack((ubq_system1, ubq_system2))

    stacked_score = ScoreSystem.build_for(
        twoubq,
        {DunbrackScore},
        weights={"dunbrack_rot": 1.0, "dunbrack_rotdev": 2.0, "dunbrack_semirot": 3.0},
    )

    coords = coords_for(twoubq, stacked_score)
    tot = stacked_score.intra_total(coords)

    forward = stacked_score.intra_forward(coords)
    assert len(forward) == 3
    for terms in forward.values():
        assert len(terms) == 2

    sumtot = torch.sum(tot)
    sumtot.backward()
