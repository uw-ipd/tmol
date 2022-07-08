from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.omega import OmegaScore
from tmol.score.modules.coords import coords_for
from tmol.system.score_support import score_method_to_even_weights_dict


def test_omega_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system, {OmegaScore}, score_method_to_even_weights_dict(OmegaScore)
    )
    coords = coords_for(ubq_system, score_system)
    intra_container = score_system.intra_forward(coords)
    assert float(intra_container["omega"]) == approx(6.741275, rel=1e-3)
