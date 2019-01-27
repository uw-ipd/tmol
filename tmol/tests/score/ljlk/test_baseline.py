import pytest
from pytest import approx

from tmol.utility.reactive import reactive_attrs

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph, LKScoreGraph


@reactive_attrs
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


@reactive_attrs
class LKGraph(CartesianAtomicCoordinateProvider, LKScoreGraph):
    pass


comparisons = {
    "lj_numpyros": pytest.param(
        LJGraph, {"total_lj": -425.3 + 248.8}, marks=pytest.mark.xfail
    ),
    "lk_numpyros": pytest.param(LKGraph, {"total_lk": 255.8}, marks=pytest.mark.xfail),
    "lj_regression": (LJGraph, {"total_lj": -177.1}),
    "lk_regression": (LKGraph, {"total_lk": 248.2}),
}


@pytest.mark.parametrize(
    "graph_class,expected_scores",
    list(comparisons.values()),
    ids=list(comparisons.keys()),
)
def test_baseline_comparison(ubq_system, torch_device, graph_class, expected_scores):
    test_graph = graph_class.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()
    scores = {
        term: float(getattr(intra_container, term).detach()) for term in expected_scores
    }

    assert scores == approx(expected_scores, rel=1e-3)
