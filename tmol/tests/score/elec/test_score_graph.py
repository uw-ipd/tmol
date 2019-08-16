import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.elec import ElecScoreGraph
from tmol.score.device import TorchDevice

from tmol.score.score_graph import score_graph


@score_graph
class ElecGraph(CartesianAtomicCoordinateProvider, ElecScoreGraph, TorchDevice):
    pass


def test_elec_w_twoubq_stacks(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    elec_graph = ElecGraph.build_for(twoubq, device=torch_device)
    tot = elec_graph.intra_score().total
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    # smoke
    torch.sum(tot).backward()


def test_jagged_scoring(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = ElecGraph.build_for(ubq40, device=torch_device)
    score60 = ElecGraph.build_for(ubq60, device=torch_device)
    score_both = ElecGraph.build_for(twoubq, device=torch_device)

    total40 = score40.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total

    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
