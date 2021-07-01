from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.omega import OmegaScore


def test_omega_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(ubq_system, {OmegaScore}, {"omega": 1.0})

    intra_container = score_system.intra_subscore()
    assert float(intra_container.total_omega) == approx(6.741275, rel=1e-3)
