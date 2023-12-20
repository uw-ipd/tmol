import pytest
import torch
from pytest import approx

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.coords import coords_for
from tmol.io import pose_stack_from_pdb
from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm


# graph_comparisons = {
#     "lj_regression": (LJScore, {"lj": -177.242}),
#     "lk_regression": (LKScore, {"lk": 298.275}),
# }

module_comparisons = {
    "lj_regression": (LJScore, {"lj": -177.242}),
    "lk_regression": (LKScore, {"lk": 298.275}),
}


def test_baseline_comparison(ubq_pdb, default_database, torch_device):
    gold_scores = {
        "lj": -177.242,
        "lk": 298.275,
    }
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(p1.packed_block_types)
    ljlk_energy.setup_poses(p1)
    ljlk_pose_scorer = ljlk_energy.render_whole_pose_scoring_module(p1)
    coords = torch.nn.Parameter(p1.coords.clone())
    tscores = ljlk_pose_scorer(coords)
    mapping = {"lj": 0, "lk": 1}
    scores = {term: tscores[term_ind, 0].item() for term, term_ind in mapping.items()}
    assert scores == approx(gold_scores, rel=1e-3)


@pytest.mark.parametrize(
    "score_method,expected_scores",
    list(module_comparisons.values()),
    ids=list(module_comparisons.keys()),
)
def test_baseline_comparison_modules(
    ubq_system, torch_device, score_method, expected_scores
):
    weight_factor = 10.0
    weights = {t: weight_factor for t in expected_scores}

    score_system: ScoreSystem = ScoreSystem.build_for(
        ubq_system,
        {score_method},
        weights=weights,
        device=torch_device,
        drop_missing_atoms=False,
    )

    coords = coords_for(ubq_system, score_system, requires_grad=False)

    scores = score_system.intra_forward(coords)

    assert {k: float(v) for k, v in scores.items()} == approx(expected_scores, rel=1e-3)

    total_score = score_system.intra_total(coords)
    assert float(total_score) == approx(
        sum(expected_scores.values()) * weight_factor, rel=1e-3
    )
