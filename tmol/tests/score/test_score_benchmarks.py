import pytest

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.score import TotalScoreGraph

from tmol.score.device import TorchDevice

from tmol.score.total_score import (
    ScoreComponentAttributes,
    TotalScoreComponentsGraph,
)

from tmol.score.bonded_atom import BondedAtomScoreGraph

from tmol.score.coordinates import CartesianAtomicCoordinateProvider, KinematicAtomicCoordinateProvider
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.hbond import HBondScoreGraph

from tmol.system.score import system_cartesian_space_graph_params, system_torsion_space_graph_params


@reactive_attrs
class DofSpaceDummy(
        KinematicAtomicCoordinateProvider,
        BondedAtomScoreGraph,
        TotalScoreComponentsGraph,
        TorchDevice,
):
    @property
    def component_total_score_terms(self):
        return ScoreComponentAttributes(
            name="dummy",
            total="dummy_total",
        )

    @reactive_property
    def dummy_total(coords):
        return coords.sum()


@reactive_attrs
class DofSpaceTotal(
        KinematicAtomicCoordinateProvider,
        TotalScoreGraph,
        TorchDevice,
):
    pass


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
    [TotalScore, DofSpaceTotal, HBondScore, LJLKScore, DofSpaceDummy],
    ids=["total_cart", "total_torsion", "hbond", "ljlk", "kinematics"],
)
@pytest.mark.parametrize(
    "benchmark_pass",
    ["full", "forward", "backward"],
)
@pytest.mark.benchmark(
    group="score_components",
)
def test_graph(
        benchmark,
        benchmark_pass,
        graph_class,
        ubq_system,
        torch_device,
):
    if issubclass(graph_class, CartesianAtomicCoordinateProvider):
        score_graph = graph_class(
            **system_cartesian_space_graph_params(
                ubq_system,
                requires_grad=True,
                device=torch_device,
            )
        )
    elif issubclass(graph_class, KinematicAtomicCoordinateProvider):
        score_graph = graph_class(
            **system_torsion_space_graph_params(
                ubq_system,
                requires_grad=True,
                device=torch_device,
            )
        )
    else:
        raise NotImplementedError

    # Score once to prep graph
    score_graph.total_score

    if benchmark_pass is "full":

        @benchmark
        def full():
            return float(score_graph.step())

    elif benchmark_pass is "forward":

        @benchmark
        def forward():
            score_graph.reset_total_score()
            float(score_graph.total_score)

    elif benchmark_pass is "backward":

        @benchmark
        def backward():
            score_graph.total_score.backward(retain_graph=True)
    else:
        raise NotImplementedError
