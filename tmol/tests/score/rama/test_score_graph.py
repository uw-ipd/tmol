import torch

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.rama import RamaScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.score_support import rama_graph_inputs
from tmol.system.packed import PackedResidueSystemStack


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph):
    pass


def test_phipsi_identification(default_database, ubq_system):
    tsys = ubq_system
    test_params = rama_graph_inputs(tsys, default_database)
    assert test_params["allphis"].shape == (1,76,5)
    assert test_params["allpsis"].shape == (1,76,5)


def test_rama_smoke(ubq_system, torch_device):
    rama_graph = RamaGraph.build_for(ubq_system, device=torch_device)
    assert rama_graph.allphis.shape == (1,76,5)
    assert rama_graph.allpsis.shape == (1,76,5)

def test_rama_w_twoubq_stacks(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    rama_graph = RamaGraph.build_for(twoubq, device=torch_device)
    tot = rama_graph.intra_score().total_rama
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    torch.sum(tot).backward()
