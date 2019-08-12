from pytest import approx


from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.cartbonded import CartBondedScoreGraph


@score_graph
class CartBondedGraph(CartesianAtomicCoordinateProvider, CartBondedScoreGraph):
    pass


def test_cartbonded_baseline_comparison(ubq_system, torch_device):
    test_graph = CartBondedGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()

    assert float(intra_container.total_cartbonded_length[0]) == approx(
        37.7848, rel=1e-3
    )
    assert float(intra_container.total_cartbonded_angle[0]) == approx(
        183.5785, rel=1e-3
    )
    assert float(intra_container.total_cartbonded_torsion[0]) == approx(
        50.5842, rel=1e-3
    )
    assert float(intra_container.total_cartbonded_improper[0]) == approx(
        9.4305, rel=1e-3
    )
    assert float(intra_container.total_cartbonded_hxltorsion[0]) == approx(
        47.4197, rel=1e-3
    )
