import pytest
from pytest import approx
import torch

from tmol.score.score_graph import score_graph

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph, LKScoreGraph

from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


@score_graph
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


@score_graph
class LKGraph(CartesianAtomicCoordinateProvider, LKScoreGraph):
    pass


comparisons = {
    "lj_regression": (LJGraph, {"total_lj": -177.1}),
    "lk_regression": (LKGraph, {"total_lk": 297.3}),
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

    # compute and synchronize
    for term in expected_scores:
        getattr(intra_container, term)
    synchronize_if_cuda_available()

    scores = {
        term: float(getattr(intra_container, term).detach()) for term in expected_scores
    }

    assert scores == approx(expected_scores, rel=1e-3)
