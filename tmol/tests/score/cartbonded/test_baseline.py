from pytest import approx

from tmol.system.score_support import score_method_to_even_weights_dict

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.coords import coords_for


def test_cartbonded_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {CartBondedScore},
        weights=score_method_to_even_weights_dict(CartBondedScore),
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )
    coords = coords_for(ubq_system, score_system)

    intra_container = score_system.intra_forward(coords)

    assert float(intra_container["cartbonded_lengths"]) == approx(37.628773, rel=1e-3)
    assert float(intra_container["cartbonded_angles"]) == approx(181.0597, rel=1e-3)
    assert float(intra_container["cartbonded_torsions"]) == approx(50.58417, rel=1e-3)
    assert float(intra_container["cartbonded_impropers"]) == approx(9.390318, rel=1e-3)
    assert float(intra_container["cartbonded_hxltorsions"]) == approx(
        47.419704, rel=1e-3
    )
