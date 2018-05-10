import torch

from tmol.score.coordinates import RealSpaceScoreGraph
from tmol.score.hbond import HBondScoreGraph

from tmol.system.residue.score import system_real_space_graph_params


class HBGraph(
        HBondScoreGraph,
        RealSpaceScoreGraph,
):
    pass


def test_hbond_smoke(ubq_system):
    hbond_graph = HBGraph(
        **system_real_space_graph_params(ubq_system, requires_grad=True)
    )

    nan_scores = torch.nonzero(torch.isnan(hbond_graph.hbond_scores))
    assert len(nan_scores) == 0
    assert (hbond_graph.total_hbond != 0).all()

    hbond_graph.total_hbond.backward()
    nan_grads = torch.nonzero(torch.isnan(hbond_graph.coords.grad))
    assert len(nan_grads) == 0


def test_hbond_smoke_bbonly(bb_hbond_database, ubq_system):
    """Backbone-only score.

    Covers cases missing specific classes of acceptors.
    """

    hbond_graph = HBGraph(
        hbond_database=bb_hbond_database,
        **system_real_space_graph_params(ubq_system, requires_grad=True)
    )

    nan_scores = torch.nonzero(torch.isnan(hbond_graph.hbond_scores))
    assert len(nan_scores) == 0
    assert (hbond_graph.total_hbond != 0).all()

    hbond_graph.total_hbond.backward()
    nan_grads = torch.nonzero(torch.isnan(hbond_graph.coords.grad))
    assert len(nan_grads) == 0
