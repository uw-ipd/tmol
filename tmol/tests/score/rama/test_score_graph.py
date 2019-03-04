import copy

import pytest
import torch

from tmol.database import ParameterDatabase

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.rama.score_graph import RamaScoreGraph
from tmol.score.device import TorchDevice

from tmol.score.score_graph import score_graph

import tmol.system.restypes as restypes
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import rama_graph_inputs


@score_graph
class RamaGraph(CartesianAtomicCoordinateProvider, RamaScoreGraph, TorchDevice):
    pass


def test_phipsi_identification(default_database, ubq_system):
    tsys = ubq_system
    test_params = rama_graph_inputs(tsys, default_database)
    assert test_params["phis"].shape[0] == 76
    assert test_params["psis"].shape[0] == 76


def test_rama_smoke(ubq_system, torch_device):
    rama_graph = RamaGraph.build_for(ubq_system, device=torch_device)
    assert rama_graph.phis.shape[0] == 76
    assert rama_graph.psis.shape[0] == 76
