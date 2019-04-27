import torch

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import CartesianEnergyNetwork, TorsionalEnergyNetwork

from tmol.score.score_graph import score_graph
from tmol.score import TotalScoreGraph
from tmol.score.device import TorchDevice

from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)


@score_graph
class TotalXyzScore(CartesianAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


@score_graph
class TotalDofScore(KinematicAtomicCoordinateProvider, TotalScoreGraph, TorchDevice):
    pass


def test_cart_network_min(ubq_system, torch_device):
    score_graph = TotalXyzScore.build_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    # score
    score_graph.intra_score().total
    model = CartesianEnergyNetwork(score_graph)
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    # score once to initialize
    E0 = score_graph.intra_score().total

    def closure():
        optimizer.zero_grad()
        score_graph.reset_coords()  # this line is necessary!

        E = model()
        E.backward()
        return E

    optimizer.step(closure)
    E1 = score_graph.intra_score().total
    assert E1 < E0


def test_torsion_network_min(ubq_system, torch_device):
    score_graph = TotalDofScore.build_for(
        ubq_system, requires_grad=True, device=torch_device
    )

    # score
    score_graph.intra_score().total
    model = TorsionalEnergyNetwork(score_graph)
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    # score once to initialize
    E0 = score_graph.intra_score().total

    def closure():
        optimizer.zero_grad()
        score_graph.reset_coords()  # this line is necessary!

        E = model()
        E.backward()
        return E

    optimizer.step(closure)
    E1 = score_graph.intra_score().total
    assert E1 < E0
