import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph, LKScoreGraph

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


@score_graph
class LKGraph(CartesianAtomicCoordinateProvider, LKScoreGraph):
    pass


def save_intermediate_grad(var):
    def store_grad(grad):
        var.grad = grad

    var.register_hook(store_grad)


def test_lj_nan_prop(ubq_system, torch_device):
    """LJ graph filters nan-coords, prevening nan entries on backward prop."""
    lj_graph = LJGraph.build_for(ubq_system, requires_grad=True, device=torch_device)

    intra_graph = lj_graph.intra_score()

    save_intermediate_grad(intra_graph.total_lj)

    intra_graph.total.backward(retain_graph=True)

    assert (intra_graph.total != 0).all()

    lj_nan_scores = torch.nonzero(torch.isnan(intra_graph.total_lj))
    lj_nan_grads = torch.nonzero(torch.isnan(intra_graph.total_lj.grad))
    assert len(lj_nan_scores) == 0
    assert len(lj_nan_grads) == 0
    assert (intra_graph.total_lj != 0).all()

    nan_coord_grads = torch.nonzero(torch.isnan(lj_graph.coords.grad))
    assert len(nan_coord_grads) == 0


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    graph_params = LJGraph.init_parameters_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    @benchmark
    def score_graph():
        score_graph = LJGraph(**graph_params)

        # Non-coordinate dependendent components for scoring
        score_graph.ljlk_atom_types

        return score_graph

    # TODO fordas add test assertions


def test_ljlk_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.ljlk)

    src: LJGraph = LJGraph.build_for(ubq_system)
    assert src.ljlk_database is ParameterDatabase.get_default().scoring.ljlk

    # Parameter database is overridden via kwarg
    src: LJGraph = LJGraph.build_for(ubq_system, ljlk_database=clone_db)
    assert src.ljlk_database is clone_db

    # Parameter database is referenced on clone
    clone: LJGraph = LJGraph.build_for(src)
    assert clone.ljlk_database is src.ljlk_database

    # Parameter database is overriden on clone via kwarg
    clone: LJGraph = LJGraph.build_for(
        src, ljlk_database=ParameterDatabase.get_default().scoring.ljlk
    )
    assert clone.ljlk_database is not src.ljlk_database
    assert clone.ljlk_database is ParameterDatabase.get_default().scoring.ljlk


def test_lj_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    lj_graph = LJGraph.build_for(twoubq)
    intra = lj_graph.intra_score()
    tot = intra.total_lj.cpu()

    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()


def test_lk_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    lk_graph = LKGraph.build_for(twoubq)
    intra = lk_graph.intra_score()
    tot = intra.total_lk.cpu()

    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
