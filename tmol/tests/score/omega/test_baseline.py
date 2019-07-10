from pytest import approx


from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.omega import OmegaScoreGraph

from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


@score_graph
class OmegaGraph(CartesianAtomicCoordinateProvider, OmegaScoreGraph):
    pass


def test_omega_baseline_comparison(ubq_system, torch_device):
    test_graph = OmegaGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()

    tot = intra_container.total_omega
    synchronize_if_cuda_available()

    assert float(tot) == approx(6.741275, rel=1e-3)
