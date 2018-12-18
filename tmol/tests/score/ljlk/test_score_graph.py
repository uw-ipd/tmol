import copy

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.ljlk import LJScoreGraph

from tmol.utility.reactive import reactive_attrs


@reactive_attrs
class LJGraph(CartesianAtomicCoordinateProvider, LJScoreGraph):
    pass


def save_intermediate_grad(var):
    def store_grad(grad):
        var.grad = grad

    var.register_hook(store_grad)


def test_lj_nan_prop(ubq_system, torch_device):
    """LJ graph filters nan-coords, prevening nan entries on backward prop."""
    try:
        lj_graph = LJGraph.build_for(
            ubq_system, requires_grad=True, device=torch_device
        )
    except AssertionError:
        # TODO: Reenable LJScoreGraph does not support cuda
        if torch_device.type == "cuda":
            pytest.xfail()
        raise

    intra_graph = lj_graph.intra_score()

    save_intermediate_grad(intra_graph.lj[1])
    # save_intermediate_grad(intra_graph.lk[1])

    intra_graph.total.backward(retain_graph=True)

    assert (intra_graph.total != 0).all()

    lj_nan_scores = torch.nonzero(torch.isnan(intra_graph.lj[1]))
    lj_nan_grads = torch.nonzero(torch.isnan(intra_graph.lj[1].grad))
    assert len(lj_nan_scores) == 0
    assert len(lj_nan_grads) == 0
    assert (intra_graph.total_lj != 0).all()

    nan_coord_grads = torch.nonzero(torch.isnan(lj_graph.coords.grad))
    assert len(nan_coord_grads) == 0


@pytest.mark.benchmark(group="score_setup")
def test_lj_score_setup(benchmark, ubq_system, torch_device):
    try:
        graph_params = LJGraph.init_parameters_for(
            ubq_system, requires_grad=True, device=torch_device
        )
    except AssertionError:
        if torch_device.type == "cuda":
            # TODO: Reenable LJScoreGraph does not support cuda
            pytest.xfail()
        raise

    @benchmark
    def score_graph():
        score_graph = LJGraph(**graph_params)

        # Non-coordinate dependendent components for scoring
        score_graph.lj_atom_types

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
