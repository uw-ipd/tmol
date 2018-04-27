import tmol.database

from tmol.score.hbond import HBondScoreGraph


def test_hbond_smoke(ubq_system):
    hbond_graph = HBondScoreGraph(
        **tmol.score.system_graph_params(ubq_system, requires_grad=False)
    )

    hbond_graph.hbond_scores
