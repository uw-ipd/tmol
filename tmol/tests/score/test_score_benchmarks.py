import pytest

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.score import TotalScoreGraph

from tmol.score.device import TorchDevice


from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.score_components import (
    ScoreComponent,
    ScoreComponentClasses,
    IntraScore,
)

from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.hbond import HBondScoreGraph


@reactive_attrs
class DummyIntra(IntraScore):
    @reactive_property
    def total_dummy(target):
        return target.coords.sum()


@reactive_attrs
class DofSpaceDummy(
    KinematicAtomicCoordinateProvider, BondedAtomScoreGraph, ScoreComponent, TorchDevice
):
    total_score_components = [
        ScoreComponentClasses("dummy", intra_container=DummyIntra, inter_container=None)
    ]


@reactive_attrs
class DofSpaceTotal(KinematicAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@reactive_attrs
class TotalScore(CartesianAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@reactive_attrs
class HBondScore(CartesianAtomicCoordinateProvider, HBondScoreGraph, TorchDevice):
    pass


@reactive_attrs
class LJLKScore(
    CartesianAtomicCoordinateProvider,
    BlockedInteratomicDistanceGraph,
    LJLKScoreGraph,
    TorchDevice,
):
    pass


def benchmark_score_pass(benchmark, score_graph, benchmark_pass):
    # Score once to prep graph
    total = score_graph.intra_score().total

    if benchmark_pass is "full":

        @benchmark
        def run():
            score_graph.reset_coords()

            total = score_graph.intra_score().total
            total.backward()

            float(total)

            return total

    elif benchmark_pass is "forward":

        @benchmark
        def run():
            score_graph.reset_coords()

            total = score_graph.intra_score().total

            float(total)

            return total

    elif benchmark_pass is "backward":

        @benchmark
        def run():
            total.backward(retain_graph=True)
            return total

    else:
        raise NotImplementedError

    return run


@pytest.mark.parametrize(
    "graph_class",
    [TotalScore, DofSpaceTotal, HBondScore, LJLKScore, DofSpaceDummy],
    ids=["total_cart", "total_torsion", "hbond", "ljlk", "kinematics"],
)
@pytest.mark.parametrize("benchmark_pass", ["full", "forward", "backward"])
@pytest.mark.benchmark(group="score_components")
def test_end_to_end_score_graph(
    benchmark, benchmark_pass, graph_class, ubq_system, torch_device
):
    score_graph = graph_class.build_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    run = benchmark_score_pass(benchmark, score_graph, benchmark_pass)

    assert run.device == torch_device
