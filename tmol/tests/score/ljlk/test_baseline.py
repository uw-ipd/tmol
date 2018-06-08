import pytest
import numpy

from tmol.utility.reactive import reactive_attrs

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph


@reactive_attrs
class LJLKGraph(
        CartesianAtomicCoordinateProvider,
        BlockedInteratomicDistanceGraph,
        LJLKScoreGraph,
):
    pass


@pytest.mark.xfail
def test_ljlk_numpyros_comparison(ubq_system):
    expected_scores = {
        'lj_atr': -425.3,
        'lj_rep': 248.8,
        'lk': 255.8,
    }

    test_graph = LJLKGraph.build_for(
        ubq_system,
        drop_missing_atoms=False,
        requires_grad=False,
    )

    numpy.testing.assert_allclose(
        test_graph.total_lj.detach(),
        expected_scores["lj_atr"] + expected_scores["lj_rep"],
        rtol=5e-3
    )

    numpy.testing.assert_allclose(
        test_graph.total_lk.detach(), expected_scores["lk"], rtol=5e-3
    )


def test_baseline_comparison(ubq_system, torch_device):
    test_graph = LJLKGraph.build_for(
        ubq_system,
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )

    expected_scores = {
        'total_lj': -176.5,
        'total_lk': 249.3,
    }

    for term, val in expected_scores.items():
        numpy.testing.assert_allclose(
            getattr(test_graph, term).detach(), val, rtol=5e-3
        )
