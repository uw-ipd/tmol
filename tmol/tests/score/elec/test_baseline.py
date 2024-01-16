from pytest import approx

from tmol.system.score_support import score_method_to_even_weights_dict

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.elec import ElecScore
from tmol.score.modules.coords import coords_for


def test_elec_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {ElecScore},
        weights=score_method_to_even_weights_dict(ElecScore),
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )
    coords = coords_for(ubq_system, score_system)

    intra_container = score_system.intra_forward(coords)

    assert intra_container["elec"].detach().cpu().numpy() == approx(
        -136.45409, rel=1e-3
    )
