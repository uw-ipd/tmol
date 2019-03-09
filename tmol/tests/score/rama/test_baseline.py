from pytest import approx


from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.rama import RamaScoreGraph


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph):
    pass


def test_cartbonded_baseline_comparison(ubq_system, torch_device):
    test_graph = RamaGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    print(type(test_graph.allphis))

    intra_container = test_graph.intra_score()
    assert float(intra_container.total_rama) == approx(-12.743369, rel=1e-3)
