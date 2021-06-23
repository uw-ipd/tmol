import torch

from tmol.tests.torch import requires_cuda

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.cartbonded import CartBondedScore


@requires_cuda
def test_cart_cuda(benchmark, ubq_system):
    score_system = ScoreSystem.build_for(
        ubq_system, {CartBondedScore}, weights={"cartbonded": 1.0}
    )

    # Score once to prep graph
    torch.cuda.nvtx.range_push("benchmark-setup")
    total = score_system.intra_score().total
    total.backward()
    torch.cuda.nvtx.range_pop()

    score_system.reset_coords()

    torch.cuda.nvtx.range_push("benchmark-forward")
    total = score_system.intra_score().total
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("benchmark-backward")
    total.backward()
    torch.cuda.nvtx.range_pop()

    assert total.device.type == "cuda"
