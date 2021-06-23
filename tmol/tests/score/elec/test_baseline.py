from pytest import approx


from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.elec import ElecScore


def test_elec_baseline_comparison(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {ElecScore},
        weights={"elec": 1.0},
        drop_missing_atoms=False,
        requires_grad=False,
        device=torch_device,
    )

    intra_container = score_system.intra_score()

    score = test_graph.intra_score().total_elec
    assert float(score) == approx(-131.9225, rel=1e-3)
