from pytest import approx


from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.cartbonded import CartBondedScoreGraph
from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


@score_graph
class CartBondedGraph(CartesianAtomicCoordinateProvider, CartBondedScoreGraph):
    pass


def test_cartbonded_baseline_comparison(ubq_system, torch_device):
    test_graph = CartBondedGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()

    # ask for all the values to be computed
    intra_container.total_cartbonded_length
    intra_container.total_cartbonded_angle
    intra_container.total_cartbonded_torsion
    intra_container.total_cartbonded_improper
    intra_container.total_cartbonded_hxltorsion

    # make sure the computation has completed
    synchronize_if_cuda_available()

    assert float(intra_container.total_cartbonded_length) == approx(37.7848, rel=1e-3)
    assert float(intra_container.total_cartbonded_angle) == approx(183.5785, rel=1e-3)
    assert float(intra_container.total_cartbonded_torsion) == approx(50.5842, rel=1e-3)
    assert float(intra_container.total_cartbonded_improper) == approx(9.4305, rel=1e-3)
    assert float(intra_container.total_cartbonded_hxltorsion) == approx(
        47.4197, rel=1e-3
    )
