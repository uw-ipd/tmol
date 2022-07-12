import torch

from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import get_full_score_system_for
from tmol.score.modules.coords import coords_for

from tmol.tests.autograd import gradcheck


def test_real_space_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = get_full_score_system_for(test_system, device=torch.device("cpu"))

    coords = coords_for(test_system, real_space)
    start_coords = coords

    def total_score(coords):
        return real_space.intra_total(coords)

    # fd this test needs work...
    gradcheck(total_score, (start_coords,), eps=1e-2, atol=5e-2, nfail=0)
