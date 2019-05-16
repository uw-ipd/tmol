import torch

from tmol.tests.torch import requires_cuda

from tmol.score.score_graph import score_graph
from tmol.score.device import TorchDevice

from tmol.score.coordinates import CartesianAtomicCoordinateProvider

from tmol.score.cartbonded import CartBondedScoreGraph


@score_graph
class CartBondedScore(
    CartesianAtomicCoordinateProvider, CartBondedScoreGraph, TorchDevice
):
    pass


@requires_cuda
def test_cart_cuda(benchmark, ubq_system):
    score_graph = CartBondedScore.build_for(
        ubq_system, requires_grad=True, device=torch.device("cuda")
    )

    # Score once to prep graph
    torch.cuda.nvtx.range_push("benchmark-setup")
    total = score_graph.intra_score().total
    total.backward()
    torch.cuda.nvtx.range_pop()

    score_graph.reset_coords()

    torch.cuda.nvtx.range_push("benchmark-forward")
    total = score_graph.intra_score().total
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("benchmark-backward")
    total.backward()
    torch.cuda.nvtx.range_pop()

    assert total.device.type == "cuda"
