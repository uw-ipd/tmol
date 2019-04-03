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


def skip_test_dunbrack_score_graph_smoke(ubq_system, default_database, torch_device):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )


def skip_test_dunbrack_score_setup(ubq_system, default_database, torch_device):
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


def test_dunbrack_score(ubq_system, torch_device, default_database):
    dunbrack_graph = CartDunbrackGraph.build_for(
        ubq_system, device=torch_device, parameter_database=default_database
    )
    intra_graph = dunbrack_graph.intra_score()
    e_dun = intra_graph.dun
    rotE, devpenE, semiE = (tensor.cpu().detach().numpy() for tensor in e_dun)

    gold_rotprobE = numpy.array(
        [
            4.0631204,
            0.57853216,
            0.25713605,
            2.4553306,
            0.0931158,
            0.21046177,
            0.0395051,
            3.1906276,
            0.25522172,
            1.6757183,
            1.5920585,
            1.9928962,
            0.08709037,
            0.47414574,
            0.35997432,
            0.11659218,
            1.0613708,
            0.08076078,
            2.24575,
            3.2775142,
            1.3433311,
            4.928035,
            0.48454452,
            0.21207313,
            0.47773707,
            2.9916263,
            1.3535863,
            1.4438618,
            3.6010082,
            0.89633393,
            4.7010255,
            0.09709472,
            0.3675585,
            0.3952158,
            0.5400303,
            4.9549565,
            1.3462752,
            0.20789276,
            0.57167846,
            0.9032417,
            0.6592185,
            3.06059,
            3.085055,
            3.4561768,
            4.464603,
        ],
        dtype="float32",
    )
    gold_devpenE = numpy.array(
        [
            2.2019336e+00,
            1.9770589e-01,
            1.7062329e-02,
            2.2284033e+00,
            1.5623730e-03,
            1.0206731e-01,
            8.8093299e-01,
            1.7300169e-01,
            1.8644860e-02,
            7.6692658e-03,
            1.6453832e-02,
            2.2262937e-01,
            1.0486361e-01,
            1.3324111e+00,
            6.3335113e-02,
            3.0330508e+00,
            1.5432676e+00,
            8.2827628e-02,
            5.1791597e-02,
            1.8852227e+00,
            4.7766706e-01,
            1.5125220e-01,
            5.2053146e+01,
            3.5261641e+00,
            3.8172710e-01,
            9.0749483e+00,
            6.4077797e+00,
            7.0618308e-01,
            5.9609115e-01,
            6.3117541e-02,
            8.5687655e-01,
            8.5730326e-01,
            8.4143192e-02,
            1.9782026e-01,
            3.0105633e-01,
            6.7469482e+00,
            6.9564277e-01,
            4.1652977e-02,
            1.6748179e-01,
            2.7447829e-01,
            3.8219827e-01,
            2.3674269e-01,
            3.4214351e-01,
            1.7866704e-01,
            2.0191493e-02,
            3.3157870e-01,
            4.7689737e-03,
            3.5549246e-03,
            1.0701549e+00,
            6.0844934e-01,
            3.5605950e+00,
            1.1795300e+00,
            3.3749375e-01,
            8.1586339e-02,
            1.6166082e-01,
            2.2054541e-01,
            2.9661820e+00,
            6.9153512e-01,
            5.2183652e+00,
            1.3278717e-01,
            8.7795454e-01,
            8.0672267e-05,
            5.2009654e-01,
            1.2237042e+00,
            7.2749794e-01,
            4.7833872e-01,
            8.0385596e-01,
            6.2933105e-01,
            3.3151742e-02,
            3.4686133e-01,
            7.0412594e-01,
            1.2192453e+01,
            9.3674831e-02,
            2.4320444e-02,
            9.4948375e-01,
            1.2893914e-04,
            3.8779364e+00,
            5.1492799e-02,
            2.1210913e-01,
            1.8141593e+00,
            3.9632455e-01,
            6.6038716e-01,
            6.8165082e-01,
            3.7212095e-01,
            6.0039390e-02,
            2.3802772e-02,
            1.0601709e-01,
            4.2291856e+00,
            2.4123597e-01,
            1.4563473e-01,
            4.2016461e-04,
            1.4158417e+00,
            9.4430703e-01,
            4.5358887e-01,
            6.3602781e+00,
            3.6172849e-01,
            1.9457538e-01,
            2.6063293e-02,
            3.3896239e+00,
            2.7757893e+00,
            1.1194772e-01,
            9.1943032e-01,
            4.7581565e-01,
            3.1450403e-01,
            1.6285714e-02,
            7.5589783e-02,
            1.5453130e-01,
            1.4294289e+00,
            3.0082038e-01,
            1.8080359e+00,
            1.1556238e+00,
            1.9699725e+00,
            9.9075073e-01,
            1.8308721e+00,
            1.3792585e-01,
            7.6053184e-01,
            1.2563300e+01,
            6.9308120e-01,
            1.1976743e-03,
            2.3833122e+00,
            1.2804033e+00,
            1.1074789e-01,
            2.5224197e-01,
            8.8699259e-02,
            9.1802626e-04,
            3.8788509e+00,
            1.2957443e+00,
            1.8763783e+01,
            2.5348916e+00,
            6.2426653e+00,
            3.4420356e-01,
            7.5524843e-01,
            1.8519447e+00,
            5.9140902e+00,
            1.3720540e+00,
            2.3464854e+00,
            1.2296946e+00,
        ],
        dtype="float32",
    )
    gold_semirotprobE = numpy.array(
        [
            5.529063,
            2.644885,
            5.000137,
            4.07152,
            2.620114,
            5.4592676,
            3.4829552,
            4.114806,
            4.1554604,
            3.7884424,
            8.816301,
            6.1051917,
            4.5643997,
            2.6311593,
            5.652759,
            4.0936203,
            2.6811717,
            2.4568348,
            2.6729317,
            6.104657,
            5.9161334,
            4.119181,
            2.9799125,
        ],
        dtype="float32",
    )

    numpy.testing.assert_almost_equal(rotE, gold_rotprobE, 1e-5)
    numpy.testing.assert_almost_equal(devpenE, gold_devpenE, 1e-5)
    numpy.testing.assert_almost_equal(semiE, gold_semirotprobE, 1e-5)


def test_cartesian_space_rama_gradcheck(ubq_res, torch_device):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    real_space = CartDunbrackGraph.build_for(test_system, device=torch_device)

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


# Only run the CPU version of this test, since on the GPU
#     f1s = torch.cross(Xs, Xs - dsc_dx)
# creates non-zero f1s even when dsc_dx is zero everywhere
def skip_test_kinematic_space_rama_gradcheck(ubq_res):
    test_system = PackedResidueSystem.from_residues(ubq_res[:6])
    torsion_space = KinematicDunbrackGraph.build_for(test_system)

    start_dofs = torsion_space.dofs.clone()

    def total_score(dofs):
        torsion_space.dofs = dofs
        return torsion_space.intra_score().total

    # x = total_score(start_dofs)

    assert torch.autograd.gradcheck(
        total_score, (start_dofs,), eps=2e-3, rtol=5e-4, atol=5e-2
    )
