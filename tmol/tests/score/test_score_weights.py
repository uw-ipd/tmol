import torch

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for
from tmol.score.modules.ljlk import LJScore
from tmol.system.packed import PackedResidueSystem
from tmol.tests.autograd import gradcheck


def test_score_weights(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 1.0})
    coords = coords_for(ubq_system, score_system)
    total1 = score_system.intra_total(coords)

    score_system = ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 0.5})
    coords = coords_for(ubq_system, score_system)
    total2 = score_system.intra_total(coords)

    torch.isclose(total1, 2.0 * total2)


def test_score_weights_grad(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    score_system = ScoreSystem.build_for(test_system, {LJScore}, weights={"lj": 0.5})
    coords = coords_for(test_system, score_system)
    start_coords = coords

    def total_score(coords):
        return score_system.intra_total(coords)

    assert gradcheck(total_score, (start_coords,), eps=1e-3, atol=2e-3, nfail=0)
