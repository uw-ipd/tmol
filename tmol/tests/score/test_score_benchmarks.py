import pytest

from tmol.score import TotalScoreGraph

from tmol.score.device import TorchDevice

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.hbond import HBondScoreGraph

from tmol.system.residue.score import system_cartesian_space_graph_params

from tmol.utility.reactive import reactive_attrs


@reactive_attrs
class TotalScore(
        CartesianAtomicCoordinateProvider,
        TotalScoreGraph,
        TorchDevice,
):
    pass


@reactive_attrs
class HBondScore(
        CartesianAtomicCoordinateProvider,
        HBondScoreGraph,
        TorchDevice,
):
    pass


@reactive_attrs
class LJLKScore(
        CartesianAtomicCoordinateProvider,
        BlockedInteratomicDistanceGraph,
        LJLKScoreGraph,
        TorchDevice,
):
    pass


@pytest.mark.parametrize(
    "graph_class",
    [TotalScore, HBondScore, LJLKScore],
    ids=["total", "hbond", "ljlk"],
)
@pytest.mark.parametrize(
    "benchmark_pass",
    ["full", "forward", "backward"],
)
@pytest.mark.benchmark(
    group="score_components",
    min_rounds=10,
    warmup=True,
    warmup_iterations=10,
)
def test_graph(
        benchmark,
        benchmark_pass,
        graph_class,
        ubq_system,
        torch_device,
):
    score_graph = graph_class(
        **system_cartesian_space_graph_params(
            ubq_system,
            requires_grad=True,
            device=torch_device,
        )
    )

    # Score once to prep graph
    score_graph.total_score

    if benchmark_pass is "full":

        @benchmark
        def full():
            score_graph.coords = score_graph.coords
            return score_graph.step()
    elif benchmark_pass is "forward":

        @benchmark
        def forward():
            score_graph.coords = score_graph.coords
            return float(score_graph.total_score)
    elif benchmark_pass is "backward":

        @benchmark
        def backward():
            score_graph.total_score.backward(retain_graph=True)
