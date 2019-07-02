import numpy
import torch

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph


@score_graph
class CartDunbrackGraph(
    CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


def test_dunbrack_score_graph_smoke(ubq_system, default_database, torch_device):
    CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def test_dunbrack_score_setup(ubq_system, default_database, torch_device):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    dun_params = dunbrack_graph.dun_resolve_indices
    aa_indices_gold = torch.tensor(
        [
            4,
            15,
            1,
            12,
            9,
            2,
            8,
            3,
            8,
            2,
            8,
            1,
            8,
            3,
            11,
            9,
            11,
            5,
            7,
            10,
            8,
            1,
            11,
            14,
            9,
            2,
            2,
            1,
            15,
            10,
            2,
            11,
            1,
            5,
            5,
            10,
            15,
            15,
            6,
            3,
            1,
            12,
            2,
            15,
            3,
            11,
            10,
            6,
            8,
            3,
            7,
            10,
            17,
            14,
            1,
            15,
            2,
            11,
            7,
            8,
            3,
            13,
            3,
            9,
            3,
            6,
            3,
            6,
        ],
        dtype=torch.int32,
    )
    assert (aa_indices_gold == dun_params.aa_indices.cpu()).all()
