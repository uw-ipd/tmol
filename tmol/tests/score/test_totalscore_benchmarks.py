import pytest

from tmol.utility.reactive import reactive_property

from tmol.score import TotalScoreGraph

from tmol.score.score_graph import score_graph
from tmol.score.device import TorchDevice
from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.score_components import ScoreComponentClasses, IntraScore

from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)


@score_graph
class TotalScore(CartesianAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@pytest.mark.benchmark(group="total_score_setup")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_setup(benchmark, systems_bysize, system_size, torch_device):
    @benchmark
    def setup():
        score_graph = TotalScore.build_for(
            systems_bysize[system_size], requires_grad=True, device=torch_device
        )
        total = score_graph.intra_score().total

    setup


@pytest.mark.benchmark(group="total_score_onepass")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_full(benchmark, systems_bysize, system_size, torch_device):
    score_graph = TotalScore.build_for(
        systems_bysize[system_size], requires_grad=True, device=torch_device
    )
    total = score_graph.intra_score().total

    @benchmark
    def forward_backward():
        score_graph.reset_coords()
        total = score_graph.intra_score().total
        total.backward()
        return total

    forward_backward
