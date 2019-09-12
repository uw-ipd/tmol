import pytest
import torch

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

    # cpu_device = torch.device("cpu")

    score40 = ElecGraph.build_for(ubq40, device=torch_device)
    score60 = ElecGraph.build_for(ubq60, device=torch_device)
    score_both = ElecGraph.build_for(twoubq, device=torch_device)

    elec40 = score40.elec_partial_charges
    elec60 = score60.elec_partial_charges
    elec_both = score_both.elec_partial_charges

    torch.testing.assert_allclose(elec40, elec60[:, : elec40.shape[1]])
    torch.testing.assert_allclose(elec60, elec_both[1:2, : elec60.shape[1]])

    torch.testing.assert_allclose(
        score40.coords, score60.coords[:, : score40.coords.shape[1]]
    )
    torch.testing.assert_allclose(
        score40.coords, score_both.coords[0:1, : score40.coords.shape[1]]
    )
    torch.testing.assert_allclose(score60.coords, score_both.coords[1:2])

    total40 = score40.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total

    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())


def test_jagged_scoring2(ubq_res, default_database, torch_device):
    ubq1050 = PackedResidueSystem.from_residues(ubq_res[10:50])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq1050, ubq60))

    score1050 = ElecGraph.build_for(ubq1050, device=torch_device)
    score60 = ElecGraph.build_for(ubq60, device=torch_device)
    score_both = ElecGraph.build_for(twoubq, device=torch_device)

    total1050 = score1050.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total

    assert total_both[0].item() == pytest.approx(total1050[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
