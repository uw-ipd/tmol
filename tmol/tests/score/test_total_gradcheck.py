import torch

from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import get_full_score_system_for

from tmol.tests.autograd import gradcheck


def test_real_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    real_space = get_full_score_system_for(test_system)

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        return real_space.intra_score_only(state_coords)

    # fd this test needs work...
    assert gradcheck(total_score, (start_coords,), eps=1e-2, atol=5e-2, nfail=0)
