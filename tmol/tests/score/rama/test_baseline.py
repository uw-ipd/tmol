from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.rama import RamaScore
from tmol.score.modules.coords import coords_for
from tmol.system.score_support import score_method_to_even_weights_dict


def test_rama_baseline_comparison(ubq_system, torch_device):
    test_system = ScoreSystem.build_for(
        ubq_system, {RamaScore}, score_method_to_even_weights_dict(RamaScore)
    )
    coords = coords_for(ubq_system, test_system)
    intra_container = test_system.intra_forward(coords)
    assert float(intra_container["rama"]) == approx(-12.743369, rel=1e-3)
