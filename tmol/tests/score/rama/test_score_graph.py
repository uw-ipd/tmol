import torch
import pytest

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.rama import RamaScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.score_support import rama_graph_inputs
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph):
    pass


def test_phipsi_identification(default_database, ubq_system):
    tsys = ubq_system
    test_params = rama_graph_inputs(tsys, default_database)
    assert test_params["allphis"].shape == (1, 76, 5)
    assert test_params["allpsis"].shape == (1, 76, 5)


def test_rama_smoke(ubq_system, torch_device):
    rama_graph = RamaGraph.build_for(ubq_system, device=torch_device)
    assert rama_graph.allphis.shape == (1, 76, 5)
    assert rama_graph.allpsis.shape == (1, 76, 5)
    

def test_rama_w_twoubq_stacks(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    rama_graph = RamaGraph.build_for(twoubq, device=torch_device)
    tot = rama_graph.intra_score().total_rama
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    torch.sum(tot).backward()


def test_jagged_scoring(ubq_res, default_database):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = RamaGraph.build_for(ubq40)
    score60 = RamaGraph.build_for(ubq60)
    score_both = RamaGraph.build_for(twoubq)
    
    total40 = score40.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total
    
    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
