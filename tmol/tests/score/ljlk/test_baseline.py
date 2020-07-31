import pytest
from pytest import approx

from tmol.score.score_graph import score_graph

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph, LKScoreGraph

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.coords import coords_for


@score_graph
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


@score_graph
class LKGraph(CartesianAtomicCoordinateProvider, LKScoreGraph):
    pass


graph_comparisons = {
    "lj_regression": (LJGraph, {"total_lj": -177.1}),
    "lk_regression": (LKGraph, {"total_lk": 297.3}),
}

module_comparisons = {
    "lj_regression": (LJScore, {"lj": -177.1}),
    "lk_regression": (LKScore, {"lk": 297.3}),
}


@pytest.mark.parametrize(
    "graph_class,expected_scores",
    list(graph_comparisons.values()),
    ids=list(graph_comparisons.keys()),
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


@pytest.mark.parametrize(
    "score_method,expected_scores",
    list(module_comparisons.values()),
    ids=list(module_comparisons.keys()),
)
def test_baseline_comparison_modules(
    ubq_system, torch_device, score_method, expected_scores
):
    weight_factor = 10.0
    weights = {t: weight_factor for t in expected_scores}

    score_system: ScoreSystem = ScoreSystem.build_for(
        ubq_system,
        {score_method},
        weights=weights,
        device=torch_device,
        drop_missing_atoms=False,
    )

    coords = coords_for(ubq_system, score_system, requires_grad=False)

    scores = score_system.intra_forward(coords)

    assert {k: float(v) for k, v in scores.items()} == approx(expected_scores, rel=1e-3)

    total_score = score_system.intra_total(coords)
    assert float(total_score) == approx(
        sum(expected_scores.values()) * weight_factor, rel=1e-3
    )
