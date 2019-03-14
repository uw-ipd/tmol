import attr
import pandas

import numpy
import torch

import itertools

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph
from tmol.types.torch import Tensor


@score_graph
class DunbrackGraph(CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice):
    pass


def temp_skip_test_dunbrack_score_graph_smoke(
    ubq_system, default_database, torch_device
):
    dunbrack_graph = DunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def temp_skip_test_dunbrack_score_setup(ubq_system, default_database, torch_device):
    dunbrack_graph = DunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )

    dun_params = dunbrack_graph.dun_resolve_indices

    ndihe_gold = numpy.array(
        [
            5,
            5,
            4,
            4,
            3,
            6,
            3,
            4,
            3,
            6,
            3,
            4,
            3,
            4,
            5,
            3,
            5,
            5,
            3,
            4,
            3,
            4,
            5,
            4,
            3,
            6,
            6,
            4,
            5,
            4,
            6,
            5,
            4,
            5,
            5,
            4,
            5,
            5,
            6,
            4,
            4,
            4,
            6,
            5,
            4,
            5,
            4,
            6,
            3,
            4,
            3,
            4,
            4,
            4,
            4,
            5,
            6,
            5,
            3,
            3,
            4,
            4,
            4,
            3,
            4,
            6,
            4,
            6,
        ],
        dtype=int,
    )
    numpy.testing.assert_array_equal(ndihe_gold, dun_params.ndihe_for_res.cpu().numpy())


def test_dunbrack_score_cpu(ubq_system, default_database):
    device = torch.device("cpu")
    dunbrack_graph = DunbrackGraph.build_for(
        ubq_system, device=device, parameter_database=default_database
    )

    intra_graph = dunbrack_graph.intra_score()
    e_dun = intra_graph.dun
