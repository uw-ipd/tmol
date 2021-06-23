from pytest import approx

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.cartbonded import CartBondedScore


def test_cartbonded_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {CartBondedScore},
        weights={"cartbonded": 1.0},
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )

    intra_container = score_system.intra_score()

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
