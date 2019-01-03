import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.hbond import HBondScoreGraph
from tmol.score.device import TorchDevice

from tmol.utility.reactive import reactive_attrs

from tmol.tests.torch import cuda_not_implemented


@reactive_attrs
class HBGraph(CartesianAtomicCoordinateProvider, HBondScoreGraph, TorchDevice):
    pass


@cuda_not_implemented
def test_hbond_smoke(ubq_system, test_hbond_database, torch_device):
    """Hbond graph filters null atoms and unused functional groups, does not
    produce nan values in backward pass.

    Params:
        test_hbond_database:
            "bb_only" covers cases missing acceptor/donor classes.
            "default" covers base case configuration.
    """

    hbond_graph = HBGraph.build_for(
        ubq_system, device=torch_device, hbond_database=test_hbond_database
    )

    intra_graph = hbond_graph.intra_score()

    ind, score = intra_graph.hbond
    nan_scores = torch.nonzero(torch.isnan(score))
    assert len(nan_scores) == 0
    assert (intra_graph.total_hbond != 0).all()
    assert intra_graph.total.device == torch_device

    intra_graph.total_hbond.backward()
    nan_grads = torch.nonzero(torch.isnan(hbond_graph.coords.grad))
    assert len(nan_grads) == 0


@pytest.mark.benchmark(group="score_setup")
@cuda_not_implemented
def test_hbond_score_setup(benchmark, ubq_system, torch_device):
    graph_params = HBGraph.init_parameters_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    @benchmark
    def score_graph():
        score_graph = HBGraph(**graph_params)

        # Non-coordinate dependent components for scoring
        score_graph.hbond_donor_indices
        score_graph.hbond_acceptor_indices

        return score_graph

    # TODO fordas add test assertions


def test_hbond_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.hbond)

    src: HBGraph = HBGraph.build_for(ubq_system)
    assert src.hbond_database is ParameterDatabase.get_default().scoring.hbond

    # Parameter database is overridden via kwarg
    src: HBGraph = HBGraph.build_for(ubq_system, hbond_database=clone_db)
    assert src.hbond_database is clone_db

    # Parameter database is referenced on clone
    clone: HBGraph = HBGraph.build_for(src)
    assert clone.hbond_database is src.hbond_database

    # Parameter database is overriden on clone via kwarg
    clone: HBGraph = HBGraph.build_for(
        src, hbond_database=ParameterDatabase.get_default().scoring.hbond
    )
    assert clone.hbond_database is not src.hbond_database
    assert clone.hbond_database is ParameterDatabase.get_default().scoring.hbond
