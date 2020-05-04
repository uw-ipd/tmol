import copy

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.hbond import HBondScoreGraph
from tmol.score.device import TorchDevice

from tmol.score.score_graph import score_graph


@score_graph
class HBGraph(CartesianAtomicCoordinateProvider, HBondScoreGraph, TorchDevice):
    pass


def test_hbond_smoke(ubq_system, test_hbond_database, torch_device):
    """Hbond graph filters null atoms and unused functional groups, does not
    produce nan values in backward pass.

    Params:
        test_hbond_database:
            "bb_only" covers cases missing acceptor/donor classes.
            "default" covers base case configuration.
    """

    hbond_graph = HBGraph.build_for(
        ubq_system, device=torch_device, hbond_database=test_hbond_database
    )

    intra_graph = hbond_graph.intra_score()

    score = intra_graph.total_hbond
    nan_scores = torch.nonzero(torch.isnan(score))
    assert len(nan_scores) == 0
    assert (intra_graph.total_hbond != 0).all()
    assert intra_graph.total.device == torch_device

    intra_graph.total_hbond.backward()
    nan_grads = torch.nonzero(torch.isnan(hbond_graph.coords.grad))
    assert len(nan_grads) == 0


@pytest.mark.benchmark(group="score_setup")
def test_hbond_score_setup(benchmark, ubq_system, torch_device):
    graph_params = HBGraph.init_parameters_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    @benchmark
    def score_graph():
        score_graph = HBGraph(**graph_params)

        # Non-coordinate dependent components for scoring
        score_graph.hbond_donor_indices
        score_graph.hbond_acceptor_indices

        return score_graph


def test_hbond_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.hbond)

    src: HBGraph = HBGraph.build_for(ubq_system)
    assert src.hbond_database is ParameterDatabase.get_default().scoring.hbond

    # Parameter database is overridden via kwarg
    src: HBGraph = HBGraph.build_for(ubq_system, hbond_database=clone_db)
    assert src.hbond_database is clone_db

    # Parameter database is referenced on clone
    clone: HBGraph = HBGraph.build_for(src)
    assert clone.hbond_database is src.hbond_database

    # Parameter database is overriden on clone via kwarg
    clone: HBGraph = HBGraph.build_for(
        src, hbond_database=ParameterDatabase.get_default().scoring.hbond
    )
    assert clone.hbond_database is not src.hbond_database
    assert clone.hbond_database is ParameterDatabase.get_default().scoring.hbond


def test_hbond_score_gradcheck(ubq_res, torch_device):
    test_system = PackedResidueSystem.from_residues(ubq_res[:20])
    real_space = HBGraph.build_for(test_system, device=torch_device)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords
        real_space.coords = state_coords
        return real_space.intra_score().total

    assert torch.autograd.gradcheck(
        total_score, (start_coords,), eps=2e-3, rtol=5e-4, atol=5e-2
    )


def test_jagged_scoring(ubq_res, default_database):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    score40 = HBGraph.build_for(ubq40)
    score60 = HBGraph.build_for(ubq60)
    score_both = HBGraph.build_for(twoubq)

    total40 = score40.intra_score().total
    total60 = score60.intra_score().total
    total_both = score_both.intra_score().total

    assert total_both[0].item() == pytest.approx(total40[0].item())
    assert total_both[1].item() == pytest.approx(total60[0].item())
