from pytest import approx

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.lk_ball import LKBallScore
from tmol.score.modules.coords import coords_for
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import score_method_to_even_weights_dict


def test_baseline_comparison(ubq_rosetta_baseline, torch_device):
    expected_scores = {
        "lk_ball": 171.47,
        "lk_ball_iso": 421.006,
        "lk_ball_bridge": 1.578,
        "lk_ball_bridge_uncpl": 10.99,
    }

    test_system = PackedResidueSystem.from_residues(ubq_rosetta_baseline.tmol_residues)

    score_system = ScoreSystem.build_for(
        test_system, {LKBallScore}, score_method_to_even_weights_dict(LKBallScore)
    )
    coords = coords_for(test_system, score_system)

    intra_container = score_system.intra_forward(coords)
    scores = {term: float(intra_container[term]) for term in expected_scores.keys()}

    assert scores == approx(expected_scores, rel=1e-3)


def test_score_ubq(ubq_res, torch_device):
    expected_scores = {
        "lk_ball": 171.47,
        "lk_ball_iso": 421.006,
        "lk_ball_bridge": 1.578,
        "lk_ball_bridge_uncpl": 10.99,
    }

    test_system = PackedResidueSystem.from_residues(ubq_res)

    score_system = ScoreSystem.build_for(
        test_system, {LKBallScore}, score_method_to_even_weights_dict(LKBallScore)
    )
    coords = coords_for(test_system, score_system)

    intra_container = score_system.intra_forward(coords)
    scores = {term: float(intra_container[term]) for term in expected_scores.keys()}
    print("scores")
    print(scores)

    assert scores == approx(expected_scores, rel=1e-3)
