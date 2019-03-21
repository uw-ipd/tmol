import attr
import pandas

import numpy
import torch

import itertools

from tmol.system.packed import PackedResidueSystem
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from tmol.score.device import TorchDevice
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.score.score_graph import score_graph
from tmol.types.torch import Tensor


@score_graph
class CartDunbrackGraph(
    CartesianAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


@score_graph
class KinematicDunbrackGraph(
    KinematicAtomicCoordinateProvider, DunbrackScoreGraph, TorchDevice
):
    pass


def temp_skip_test_dunbrack_score_graph_smoke(
    ubq_system, default_database, torch_device
):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def temp_skip_test_dunbrack_score_setup(ubq_system, default_database, torch_device):
    dunbrack_graph = CartDunbrackGraph.build_for(
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
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=device, parameter_database=default_database
    )
    # dun_params = dunbrack_graph.dun_resolve_indices

    intra_graph = dunbrack_graph.intra_score()
    e_dun = intra_graph.dun


def test_cartesian_space_rama_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = CartDunbrackGraph.build_for(test_system)

    coord_mask = torch.isnan(real_space.coords).sum(dim=-1) == 0
    start_coords = real_space.coords[coord_mask]

    def total_score(coords):
        state_coords = real_space.coords.detach().clone()
        state_coords[coord_mask] = coords
        real_space.coords = state_coords
        return real_space.intra_score().total

    assert torch.autograd.gradcheck(
        total_score, (start_coords,), eps=2e-3, rtol=5e-2, atol=5e-2
    )


def test_kinematic_space_rama_gradcheck():
    from tmol.system.io import ResidueReader
    import os

    # print("pwd")
    # print(os.getcwd())
    ubq_res = ResidueReader.get_default().parse_pdb(
        open("1ubq_res2_n90phi_140psi_160chi3.pdb").read()
    )

    # print("ubq_res")
    # print(ubq_res[:6])
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    torsion_space = KinematicDunbrackGraph.build_for(test_system)

    # print("kinematics meta data")
    # print(test_system.torsion_metadata)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.intra_score().total

    x = total_score(start_dofs)

    # assert torch.autograd.gradcheck(
    #     total_score, (start_coords,), eps=2e-3, rtol=5e-2, atol=5e-2
    # )
