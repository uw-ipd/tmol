from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.rama import RamaScore


def test_rama_baseline_comparison(ubq_system, torch_device):
    test_graph = ScoreSystem.build_for(ubq_system, {RamaScore}, {"rama": 1.0})

    intra_container = test_graph.intra_total()
    assert float(intra_container.total_rama) == approx(-12.743369, rel=1e-3)
