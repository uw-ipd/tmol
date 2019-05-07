import pytest
from pytest import approx

from tmol.score.score_graph import score_graph

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.lk_ball.score_graph import LKBallScoreGraph

from tmol.tests.torch import cuda_not_implemented

from tmol.system.packed import PackedResidueSystem


@score_graph
class LKBallGraph(CartesianAtomicCoordinateProvider, LKBallScoreGraph):
    pass


# rosetta-baseline values:
# {
#     "lk_ball": 173.68865865110556,
#     "lk_ball_iso": 411.1702730219401,
#     "lk_ball_bridge": 1.426083767458333,
#     "lk_ball_bridge_uncpl": 10.04351344360775,
# }

comparisons = {
    "lkball_regression": (
        LKBallGraph,
        {
            "total_lk_ball": 171.47,
            "total_lk_ball_iso": 421.006,
            "total_lk_ball_bridge": 1.578,
            "total_lk_ball_bridge_uncpl": 10.99,
        },
    )
}


@pytest.mark.parametrize(
    "graph_class,expected_scores",
    list(comparisons.values()),
    ids=list(comparisons.keys()),
)
# @cuda_not_implemented
def test_baseline_comparison(
    ubq_rosetta_baseline, torch_device, graph_class, expected_scores
):
    test_system = PackedResidueSystem.from_residues(ubq_rosetta_baseline.tmol_residues)

    test_graph = graph_class.build_for(
        test_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()
    scores = {
        term: float(getattr(intra_container, term).detach()) for term in expected_scores
    }

    assert scores == approx(expected_scores, rel=1e-3)
