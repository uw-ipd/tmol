import pytest

import torch

from tmol.score.ljlk import LJScoreGraph
from tmol.score.device import TorchDevice
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.score_graph import score_graph
from tmol.score.score_weights import ScoreWeights

from tmol.system.packed import PackedResidueSystem

from tmol.tests.autograd import gradcheck


@score_graph
class LJScore(
    CartesianAtomicCoordinateProvider, LJScoreGraph, ScoreWeights, TorchDevice
):
    pass


def test_score_weights(ubq_system, torch_device):
    score_graph = LJScore.build_for(
        ubq_system,
        requires_grad=True,
        device=torch_device,
        component_weights={"total_lj": 1.0},
    )
    total1 = score_graph.intra_score().total

    score_graph = LJScore.build_for(
        ubq_system,
        requires_grad=True,
        device=torch_device,
        component_weights={"total_lj": 0.5},
    )
    total2 = score_graph.intra_score().total

    torch.isclose(total1, 2.0 * total2)


def test_score_weights_grad(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = LJScore.build_for(test_system, component_weights={"total_lj": 0.5})

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords

        real_space.coords = state_coords
        return real_space.intra_score().total

    assert gradcheck(total_score, (start_coords,), eps=1e-3, atol=1e-3, nfail=0)
