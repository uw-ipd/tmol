import torch
import pytest

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.cartbonded import CartBondedScoreGraph
from tmol.score.score_graph import score_graph
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class CartBondedGraph(CartesianAtomicCoordinateProvider, CartBondedScoreGraph):
    pass


def test_cartbonded_smoke(ubq_system, torch_device):
    cb_graph = CartBondedGraph.build_for(ubq_system, device=torch_device)
    ang = cb_graph.intra_score().total_cartbonded_angle
    assert ang.shape == (1,)


def test_cartbonded_w_twoubq_stacks(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    cb_graph = CartBondedGraph.build_for(twoubq, device=torch_device)
    tot_len = cb_graph.intra_score().total_cartbonded_length
    assert tot_len.shape == (2,)
    torch.testing.assert_allclose(tot_len[0], tot_len[1])

    # smoke
    torch.sum(tot_len).backward()


def test_jagged_scoring(ubq_res, default_database):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = CartBondedGraph.build_for(ubq40)
    score60 = CartBondedGraph.build_for(ubq60)
    score_both = CartBondedGraph.build_for(twoubq)

    total40 = score40.intra_score().total_cartbonded_length
    total60 = score60.intra_score().total_cartbonded_length
    total_both = score_both.intra_score().total_cartbonded_length

    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
