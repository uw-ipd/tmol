from pytest import approx
import torch

from tmol.score.score_graph import score_graph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.elec import ElecScoreGraph

from tmol.utility.cuda.synchronize import synchronize_if_cuda_available


@score_graph
class ElecGraph(CartesianAtomicCoordinateProvider, ElecScoreGraph):
    pass


def test_elec_baseline_comparison(ubq_system, torch_device):
    test_graph = ElecGraph.build_for(
        ubq_system, drop_missing_atoms=False, requires_grad=False, device=torch_device
    )

    score = test_graph.intra_score().total_elec
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    assert float(score) == approx(-131.9225, rel=1e-3)
