from pytest import approx


from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.elec import ElecScoreGraph


@score_graph
class ElecGraph(CartesianAtomicCoordinateProvider, ElecScoreGraph):
    pass


def test_elec_baseline_comparison(ubq_system, torch_device):
    test_graph = ElecGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()
    assert float(intra_container.total_elec) == approx(-131.9225, rel=1e-3)