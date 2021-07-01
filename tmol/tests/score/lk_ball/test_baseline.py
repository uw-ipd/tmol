import pytest
from pytest import approx

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.lk_ball import LKBallScore


def test_baseline_comparison(ubq_rosetta_baseline, torch_device):
    expected_scores = {
        "total_lk_ball": 171.47,
        "total_lk_ball_iso": 421.006,
        "total_lk_ball_bridge": 1.578,
        "total_lk_ball_bridge_uncpl": 10.99,
    }

    test_system = PackedResidueSystem.from_residues(ubq_rosetta_baseline.tmol_residues)

    score_system = ScoreSystem.build_for(test_system, {LKBallScore}, {"lk_ball": 1.0})

    intra_container = score_system.intra_subscores()
    scores = {
        term: float(getattr(intra_container, term).detach()) for term in expected_scores
    }

    assert scores == approx(expected_scores, rel=1e-3)
