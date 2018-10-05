import copy

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk.score_graph import LJLKScoreGraph
from tmol.score.ljlk.jit_score_graph import JitLJLKScoreGraph
from tmol.score.interatomic_distance import BlockedInteratomicDistanceGraph

from tmol.utility.reactive import reactive_attrs

from tmol.tests.benchmark import subfixture


@reactive_attrs
class LJLKGraph(
    CartesianAtomicCoordinateProvider, BlockedInteratomicDistanceGraph, LJLKScoreGraph
):
    pass


def save_intermediate_grad(var):
    def store_grad(grad):
        var.grad = grad

    var.register_hook(store_grad)


def test_ljlk_nan_prop(ubq_system, torch_device):
    """LJLK graph filters nan-coords, prevening nan entries on backward prop."""
    ljlk_graph = LJLKGraph.build_for(
        ubq_system, requires_grad=True, device=torch_device
    )
    save_intermediate_grad(ljlk_graph.atom_pair_dist)
    save_intermediate_grad(ljlk_graph.atom_pair_delta)

    intra_graph = ljlk_graph.intra_score()

    save_intermediate_grad(intra_graph.lj)
    save_intermediate_grad(intra_graph.lk)

    intra_graph.total.backward(retain_graph=True)

    assert (intra_graph.total != 0).all()

    lj_nan_scores = torch.nonzero(torch.isnan(intra_graph.lj))
    lj_nan_grads = torch.nonzero(torch.isnan(intra_graph.lj.grad))
    assert len(lj_nan_scores) == 0
    assert len(lj_nan_grads) == 0
    assert (intra_graph.total_lj != 0).all()

    lk_nan_scores = torch.nonzero(torch.isnan(intra_graph.lk))
    lk_nan_grads = torch.nonzero(torch.isnan(intra_graph.lk.grad))
    assert len(lk_nan_scores) == 0
    assert len(lk_nan_grads) == 0
    assert (intra_graph.total_lk != 0).all()

    nonzero_dist_grads = torch.nonzero(ljlk_graph.atom_pair_dist.grad)
    assert len(nonzero_dist_grads) != 0

    nonzero_delta_grads = torch.nonzero(ljlk_graph.atom_pair_delta.grad)
    assert len(nonzero_delta_grads) != 0
    nan_delta_grads = torch.nonzero(torch.isnan(ljlk_graph.atom_pair_delta.grad))
    assert len(nan_delta_grads) == 0

    nan_coord_grads = torch.nonzero(torch.isnan(ljlk_graph.coords.grad))
    assert len(nan_coord_grads) == 0


@pytest.mark.benchmark(group="score_setup")
def test_ljlk_score_setup(benchmark, ubq_system, torch_device):
    graph_params = LJLKGraph.init_parameters_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    @benchmark
    def score_graph():
        score_graph = LJLKGraph(**graph_params)

        # Non-coordinate dependendent components for scoring
        score_graph.ljlk_atom_pair_params
        score_graph.ljlk_bonded_path_length

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


def test_jit_graph(benchmark, torch_device, ubq_system):
    @reactive_attrs
    class LJLKGraph(
        CartesianAtomicCoordinateProvider,
        BlockedInteratomicDistanceGraph,
        LJLKScoreGraph,
    ):
        pass

    @reactive_attrs
    class JitLJLKGraph(JitLJLKScoreGraph, CartesianAtomicCoordinateProvider):
        pass

    sg = LJLKGraph.build_for(ubq_system, device=torch_device, requires_grad=False)
    jit_sg = JitLJLKGraph.build_for(
        ubq_system, device=torch_device, requires_grad=False
    )

    # Pre-load totals
    sg.intra_score().total
    jit_sg.intra_score().total

    @subfixture(benchmark)
    def naive():
        sg.reset_coords()
        return float(sg.intra_score().total_lj)

    @subfixture(benchmark)
    def jit():
        jit_sg.reset_coords()
        return float(jit_sg.intra_score().total_lj)

    torch.testing.assert_allclose(naive, jit)
