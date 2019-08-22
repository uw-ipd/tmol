import torch
import pytest

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.lk_ball import LKBallScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class LKBGraph(CartesianAtomicCoordinateProvider, LKBallScoreGraph):
    pass


def test_lkball_smoke(ubq_system, torch_device):
    lkb_graph = LKBGraph.build_for(ubq_system, device=torch_device)
    tot = lkb_graph.intra_score().total_lk_ball
    assert tot.shape == (1,)


def test_lkball_w_twoubq_stacks(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    lkb_graph = LKBGraph.build_for(twoubq, device=torch_device)
    tot = lkb_graph.intra_score().total_lk_ball
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    # smoke
    torch.sum(tot).backward()


def test_jagged_scoring(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = LKBGraph.build_for(ubq40, device=torch_device)
    score60 = LKBGraph.build_for(ubq60, device=torch_device)
    score_both = LKBGraph.build_for(twoubq, device=torch_device)

    total40 = score40.intra_score().total_lk_ball
    total60 = score60.intra_score().total_lk_ball
    total_both = score_both.intra_score().total_lk_ball

    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
