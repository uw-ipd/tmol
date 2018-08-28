import copy

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJLKScoreGraph
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

from tmol.utility.reactive import reactive_attrs


@reactive_attrs
class LJLKGraph(
    CartesianAtomicCoordinateProvider, BlockedInteratomicDistanceGraph, LJLKScoreGraph
):
    pass


def save_intermediate_grad(var):
    def store_grad(grad):
        var.grad = grad

    var.register_hook(store_grad)


def test_ljlk_smoke(ubq_system, torch_device):
    score_graph = LJLKGraph.build_for(
        ubq_system, requires_grad=True, device=torch_device
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
    nan_delta_grads = torch.nonzero(torch.isnan(score_graph.atom_pair_delta.grad))
    assert len(nan_delta_grads) == 0

    nan_coord_grads = torch.nonzero(torch.isnan(score_graph.coords.grad))
    assert len(nan_coord_grads) == 0


@pytest.mark.benchmark(group="score_setup")
def test_ljlk_score_setup(benchmark, ubq_system, torch_device):
    graph_params = LJLKGraph.init_parameters_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    @benchmark
    def score_graph():
        score_graph = LJLKGraph(**graph_params)

        # Non-coordinate depdendent components for scoring
        score_graph.ljlk_atom_pair_params
        score_graph.ljlk_interaction_weight

        return score_graph

    # TODO fordas add test assertions


def test_ljlk_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.ljlk)

    src: LJLKGraph = LJLKGraph.build_for(ubq_system)
    assert src.ljlk_database is ParameterDatabase.get_default().scoring.ljlk

    # Parameter database is overridden via kwarg
    src: LJLKGraph = LJLKGraph.build_for(ubq_system, ljlk_database=clone_db)
    assert src.ljlk_database is clone_db

    # Parameter database is referenced on clone
    clone: LJLKGraph = LJLKGraph.build_for(src)
    assert clone.ljlk_database is src.ljlk_database

    # Parameter database is overriden on clone via kwarg
    clone: LJLKGraph = LJLKGraph.build_for(
        src, ljlk_database=ParameterDatabase.get_default().scoring.ljlk
    )
    assert clone.ljlk_database is not src.ljlk_database
    assert clone.ljlk_database is ParameterDatabase.get_default().scoring.ljlk
