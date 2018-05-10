import pytest
import torch
import numpy

from tmol.system.residue.score import system_cartesian_space_graph_params

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph


class LJLKGraph(
        LJLKScoreGraph,
        BlockedInteratomicDistanceGraph,
        CartesianAtomicCoordinateProvider,
):
    pass


@pytest.mark.xfail
def test_ljlk_numpyros_comparison(ubq_system):
    test_structure = ubq_system

    test_params = system_cartesian_space_graph_params(
        test_structure,
        drop_missing_atoms=False,
        requires_grad=False,
    )

    expected_scores = {
        'lj_atr': -425.3,
        'lj_rep': 248.8,
        'lk': 255.8,
    }

    numpy.testing.assert_allclose(
        LJLKGraph(**test_params).total_lj.detach(),
        expected_scores["lj_atr"] + expected_scores["lj_rep"],
        rtol=5e-3
    )

    numpy.testing.assert_allclose(
        LJLKGraph(**test_params).total_lk.detach(),
        expected_scores["lk"],
        rtol=5e-3
    )


def test_baseline_comparison(ubq_system):
    test_structure = ubq_system

    test_params = system_cartesian_space_graph_params(
        test_structure,
        drop_missing_atoms=False,
        requires_grad=False,
    )

    test_graph = LJLKGraph(**test_params)

    expected_scores = {
        'total_lj': -176.5,
        'total_lk': 249.3,
    }

    for term, val in expected_scores.items():
        numpy.testing.assert_allclose(
            getattr(test_graph, term).detach(), val, rtol=5e-3
        )


def save_intermediate_grad(var):
    def store_grad(grad):
        var.grad = grad

    var.register_hook(store_grad)


def test_ljlk_smoke(ubq_system):
    score_graph = LJLKGraph(
        **system_cartesian_space_graph_params(ubq_system, requires_grad=True)
    )

    save_intermediate_grad(score_graph.lj)
    save_intermediate_grad(score_graph.lk)
    save_intermediate_grad(score_graph.atom_pair_dist)
    save_intermediate_grad(score_graph.atom_pair_delta)

    score_graph.total_score.backward(retain_graph=True)

    assert (score_graph.total_score != 0).all()

    lj_nan_scores = torch.nonzero(torch.isnan(score_graph.lj))
    lj_nan_grads = torch.nonzero(torch.isnan(score_graph.lj.grad))
    assert len(lj_nan_scores) == 0
    assert len(lj_nan_grads) == 0
    assert (score_graph.total_lj != 0).all()

    lk_nan_scores = torch.nonzero(torch.isnan(score_graph.lk))
    lk_nan_grads = torch.nonzero(torch.isnan(score_graph.lk.grad))
    assert len(lk_nan_scores) == 0
    assert len(lk_nan_grads) == 0
    assert (score_graph.total_lk != 0).all()

    nonzero_dist_grads = torch.nonzero(score_graph.atom_pair_dist.grad)
    assert len(nonzero_dist_grads) != 0

    nonzero_delta_grads = torch.nonzero(score_graph.atom_pair_delta.grad)
    assert len(nonzero_delta_grads) != 0
    nan_delta_grads = torch.nonzero(
        torch.isnan(score_graph.atom_pair_delta.grad)
    )
    assert len(nan_delta_grads) == 0

    nan_coord_grads = torch.nonzero(torch.isnan(score_graph.coords.grad))
    assert len(nan_coord_grads) == 0
