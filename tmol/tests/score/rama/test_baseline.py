from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.rama import RamaScore


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph):
    pass


def test_rama_baseline_comparison(ubq_system, torch_device):
    test_graph = RamaGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    intra_container = test_graph.intra_score()
    assert float(intra_container.total_rama) == approx(-12.743369, rel=1e-3)
