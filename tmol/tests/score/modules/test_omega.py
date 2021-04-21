import pytest
import torch

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.omega import OmegaScore, OmegaParameters
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    @benchmark
    def score_graph():
        return ScoreSystem.build_for(ubq_system, {OmegaScore}, weights={"lj": 1.0})


def test_omega_database_clone_factory(ubq_system):
    src = ScoreSystem._build_with_modules(ubq_system, {OmegaParameters})
    assert OmegaParameters.get(src).allomegas.shape == (1, 76, 4)

    # Allomegas is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {OmegaParameters})
    assert OmegaParameters.get(clone).allomegas is OmegaParameters.get(src).allomegas


def test_lj_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(twoubq, {OmegaScore}, weights={"omega": 1.0})

    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
