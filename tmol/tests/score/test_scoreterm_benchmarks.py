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

from tmol.score.ljlk import LJScoreGraph, LKScoreGraph
from tmol.score.hbond import HBondScoreGraph
from tmol.score.elec import ElecScoreGraph
from tmol.score.rama import RamaScoreGraph
from tmol.score.omega import OmegaScoreGraph
from tmol.score.dunbrack import DunbrackScoreGraph
from tmol.score.cartbonded import CartBondedScoreGraph
from tmol.score.lk_ball import LKBallScoreGraph


@score_graph
class DummyIntra(IntraScore):
    @reactive_property
    def total_dummy(target):
        return target.coords.sum()


@score_graph
class DofSpaceDummy(
    KinematicAtomicCoordinateProvider, BondedAtomScoreGraph, TorchDevice
):
    total_score_components = [
        ScoreComponentClasses("dummy", intra_container=DummyIntra, inter_container=None)
    ]


@score_graph
class DofSpaceTotal(KinematicAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@score_graph
class TotalScore(CartesianAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@score_graph
class HBondScore(CartesianAtomicCoordinateProvider, HBondScoreGraph, TorchDevice):
    pass


@score_graph
class ElecScore(CartesianAtomicCoordinateProvider, ElecScoreGraph, TorchDevice):
    pass


@score_graph
class RamaScore(CartesianAtomicCoordinateProvider, RamaScoreGraph, TorchDevice):
    pass


@score_graph
class OmegaScore(CartesianAtomicCoordinateProvider, OmegaScoreGraph, TorchDevice):
    pass


@score_graph
class DunbrackScore(CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice):
    pass


@score_graph
class CartBondedScore(
    CartesianAtomicCoordinateProvider, CartBondedScoreGraph, TorchDevice
):
    pass


@score_graph
class LJScore(CartesianAtomicCoordinateProvider, LJScoreGraph, TorchDevice):
    pass


@score_graph
class LKScore(CartesianAtomicCoordinateProvider, LKScoreGraph, TorchDevice):
    pass


@score_graph
class LKBallScore(CartesianAtomicCoordinateProvider, LKBallScoreGraph, TorchDevice):
    pass


def benchmark_score_pass(benchmark, score_graph, benchmark_pass):
    # Score once to prep graph
    total = score_graph.intra_score().total

    if benchmark_pass == "full":

        @benchmark
        def run():
            score_graph.reset_coords()

            total = score_graph.intra_score().total
            total.backward()

            float(total)

            return total

    elif benchmark_pass == "forward":

        @benchmark
        def run():
            score_graph.reset_coords()

            total = score_graph.intra_score().total

            float(total)

            return total

    elif benchmark_pass == "backward":

        @benchmark
        def run():
            total.backward(retain_graph=True)
            return total

    else:
        raise NotImplementedError

    return run


@pytest.mark.parametrize(
    "graph_class",
    [
        TotalScore,
        DofSpaceTotal,
        HBondScore,
        ElecScore,
        RamaScore,
        OmegaScore,
        DunbrackScore,
        CartBondedScore,
        LJScore,
        LKScore,
        LKBallScore,
        DofSpaceDummy,
    ],
    ids=[
        "total_cart",
        "total_torsion",
        "hbond",
        "elec",
        "rama",
        "omega",
        "dun",
        "cartbonded",
        "lj",
        "lk",
        "lk_ball",
        "kinematics",
    ],
)
@pytest.mark.parametrize("benchmark_pass", ["full", "forward", "backward"])
@pytest.mark.benchmark(group="score_components")
def test_end_to_end_score_graph(
    benchmark, benchmark_pass, graph_class, torch_device, ubq_system
):
    target_system = ubq_system

    score_graph = graph_class.build_for(
        target_system, requires_grad=True, device=torch_device
    )

    run = benchmark_score_pass(benchmark, score_graph, benchmark_pass)

    assert run.device == torch_device
