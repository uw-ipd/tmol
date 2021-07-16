import torch

from tmol.tests.torch import requires_cuda

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.coords import coords_for
from tmol.system.score_support import score_method_to_even_weights_dict


@requires_cuda
def test_cart_cuda(benchmark, ubq_system):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {CartBondedScore},
        weights=score_method_to_even_weights_dict(CartBondedScore),
        device=torch.device("cuda"),
    )
    coords = coords_for(ubq_system, score_system)

    # Score once to prep graph
    torch.cuda.nvtx.range_push("benchmark-setup")
    total = score_system.intra_total(coords)
    total.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("benchmark-forward")
    total = score_system.intra_total(coords)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("benchmark-backward")
    total.backward()
    torch.cuda.nvtx.range_pop()

    assert total.device.type == "cuda"
