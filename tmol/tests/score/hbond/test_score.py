import numpy
import tmol.database

from tmol.score.hbond import HBondScoreGraph


def test_hbond_smoke(ubq_system):
    hbond_graph = HBondScoreGraph(
        **tmol.score.system_graph_params(ubq_system, requires_grad=False)
    )

    hbond_scores = numpy.array(hbond_graph.hbond_scores)
    nan_scores = numpy.flatnonzero(numpy.isnan(hbond_scores))

    assert len(nan_scores) == 0
