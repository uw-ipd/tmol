import pytest
import numpy

from tmol.utility.reactive import reactive_attrs

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph


@reactive_attrs
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


def test_lj_baseline_comparison(ubq_system, torch_device):
    try:
        test_graph = LJGraph.build_for(
            ubq_system,
            drop_missing_atoms=False,
            requires_grad=False,
            device=torch_device,
        )
    except AssertionError:
        # TODO: Reenable, LJScoreGraph does not support cuda
        if torch_device.type == "cuda":
            pytest.xfail()
        raise

    expected_scores = {"total_lj": -425.3 + 248.8}

    for term, val in expected_scores.items():
        scores = test_graph.intra_score()
        numpy.testing.assert_allclose(getattr(scores, term).detach(), val, rtol=5e-3)


@pytest.mark.xfail
def test_lk_baseline_comparison(ubq_system, torch_device):
    test_graph = LKGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    expected_scores = {"total_lk": 255.8}

    for term, val in expected_scores.items():
        scores = test_graph.intra_score()
        numpy.testing.assert_allclose(getattr(scores, term).detach(), val, rtol=5e-3)
