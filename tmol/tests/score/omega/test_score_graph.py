import torch

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.omega import OmegaScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class OmegaGraph(CartesianAtomicCoordinateProvider, OmegaScoreGraph):
    pass


def test_omega_smoke(ubq_system, torch_device):
    omega_graph = OmegaGraph.build_for(ubq_system, device=torch_device)
    assert omega_graph.allomegas.shape == (1, 76, 4)


def test_jagged_scoring(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = OmegaGraph.build_for(ubq40, device=torch_device)
    score60 = OmegaGraph.build_for(ubq60, device=torch_device)
    score_both = OmegaGraph.build_for(twoubq, device=torch_device)

    total40 = score40.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total

    torch.testing.assert_allclose(total_both[0], total40[0])
    torch.testing.assert_allclose(total_both[1], total60[0])

    # smoke
    torch.sum(total_both).backward()


def test_jagged_scoring2(ubq_res, default_database, torch_device):
    ubq1050 = PackedResidueSystem.from_residues(ubq_res[10:50])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    threeubq = PackedResidueSystemStack((ubq1050, ubq60, ubq40))

    score1050 = OmegaGraph.build_for(ubq1050, device=torch_device)
    score40 = OmegaGraph.build_for(ubq40, device=torch_device)
    score60 = OmegaGraph.build_for(ubq60, device=torch_device)
    score_all = OmegaGraph.build_for(threeubq, device=torch_device)

    total1050 = score1050.intra_score().total
    total60 = score60.intra_score().total
    total40 = score40.intra_score().total
    total_all = score_all.intra_score().total

    torch.testing.assert_allclose(total_all[0], total1050[0])
    torch.testing.assert_allclose(total_all[1], total60[0])
    torch.testing.assert_allclose(total_all[2], total40[0])
