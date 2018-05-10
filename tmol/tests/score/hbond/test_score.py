import numpy

from tmol.score.coordinates import RealSpaceScoreGraph
from tmol.score.hbond import HBondScoreGraph

from tmol.system.residue.score import system_real_graph_params


class HBGraph(
        HBondScoreGraph,
        RealSpaceScoreGraph,
):
    pass


def test_hbond_smoke(ubq_system):

    hbond_graph = HBGraph(
        **system_real_graph_params(ubq_system, requires_grad=False)
    )

    hbond_scores = numpy.array(hbond_graph.hbond_scores)
    nan_scores = numpy.flatnonzero(numpy.isnan(hbond_scores))

    assert len(nan_scores) == 0


def test_hbond_smoke_bbonly(bb_hbond_database, ubq_system):
    """Backbone-only score.

    Covers cases missing specific classes of acceptors.
    """

    hbond_graph = HBGraph(
        hbond_database=bb_hbond_database,
        **system_real_graph_params(ubq_system, requires_grad=False)
    )

    hbond_scores = numpy.array(hbond_graph.hbond_scores)
    nan_scores = numpy.flatnonzero(numpy.isnan(hbond_scores))

    assert len(nan_scores) == 0
