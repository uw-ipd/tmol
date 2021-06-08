import torch

from tmol.score.modules.bases import ScoreSystem
from tmol.system.packed import PackedResidueSystem
from tmol.tests.autograd import gradcheck


def test_score_weights(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 1.0})
    total1 = score_system.intra_total()

    score_system = ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 0.5})
    total1 = score_system.intra_total()

    torch.isclose(total1, 2.0 * total2)


def test_score_weights_grad(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    score_system = ScoreSystem.build_for(ubq_system, {LJScore}, weights={"lj": 0.5})

    coord_mask = torch.isnan(score_system.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        real_space.coords = state_coords
        return real_space.intra_score().total

    assert gradcheck(total_score, (start_coords,), eps=1e-3, atol=2e-3, nfail=0)
