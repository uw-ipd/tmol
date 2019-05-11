import pytest
import math
import torch
import torch.autograd
from tmol.utility.units import parse_angle
import numpy
from tmol.tests.autograd import gradcheck
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.chemical_database import AtomTypeParamResolver


@pytest.fixture
def ljlk_params(default_database):
    return LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch.device("cpu")
    )


@pytest.fixture
def atype_params(default_database):
    return AtomTypeParamResolver.from_database(
        default_database.chemical, torch.device("cpu")
    )


def test_build_acc_waters():
    from .compiled import BuildAcceptorWater

    tensor = torch.DoubleTensor

    ## test 1: acceptor water generation + derivatives
    A = tensor((-6.071, -0.619, -3.193))
    B = tensor((-5.250, -1.595, -2.543))
    B0 = tensor((-5.489, 0.060, -3.542))

    dist = 2.65
    angle = parse_angle("109.0 deg")
    torsions = tensor([parse_angle(f"{a} deg") for a in (120.0, 240.0)])

    waters_ref = tensor(
        [[-7.42086525, -1.79165583, -5.14882262], [-7.75428876, 0.40906314, -1.4232189]]
    )

    for torsion, water_ref in zip(torsions, waters_ref):
        water = BuildAcceptorWater.apply(A, B, B0, dist, angle, torsion)
        torch.testing.assert_allclose(water, water_ref)

        gradcheck(
            lambda A, B, B0: BuildAcceptorWater.apply(A, B, B0, dist, angle, torsion),
            (A.requires_grad_(True), B.requires_grad_(True), B0.requires_grad_(True)),
        )


def test_build_don_water():
    from .compiled import BuildDonorWater

    tensor = torch.DoubleTensor

    ## test 2: donor water generation + derivatives
    D = tensor((-6.007, 4.706, -0.074))
    H = tensor((-6.747, 4.361, 0.549))

    dist = 2.65

    assert not any(t.requires_grad for t in (D, H))

    waters = BuildDonorWater.apply(D, H, dist)
    waters_ref = tensor([-7.91642236, 3.81579633, 1.5335272])
    torch.testing.assert_allclose(waters, waters_ref)

    gradcheck(
        lambda D, H: BuildDonorWater.apply(D, H, dist),
        (D.requires_grad_(True), H.requires_grad_(True)),
    )


def test_lk_fraction():
    from .compiled import LKFraction, BuildAcceptorWater

    tensor = torch.DoubleTensor
    I = dict(  # noqa
        A=tensor((0.0, 0.0, 0.0)),
        B=tensor((0.0, 0.0, -1.0)),
        B0=tensor((1.0, 0.0, 0.0)),
    )
    dist = tensor([2.65]).reshape(())
    angle = tensor([parse_angle("109.0 deg")]).reshape(())
    torsions = tensor([parse_angle(f"{a} deg") for a in (120.0, 240.0)])

    WI = torch.stack(
        [
            BuildAcceptorWater.apply(I["A"], I["B"], I["B0"], dist, angle, torsion)
            for torsion in torsions
        ]
    )

    J = tensor((-2.5, 0.1, 2.5))
    lj_radius_j = tensor([1.8]).reshape(())
    lkfrac = LKFraction.apply(WI, J, lj_radius_j)

    assert float(lkfrac) == pytest.approx(.65, abs=.01)

    gradcheck(
        lambda WI, J: LKFraction.apply(WI, J, dist),
        (WI.requires_grad_(True), J.requires_grad_(True)),
    )


def test_lk_bridge_fraction():
    from .compiled import LKBridgeFraction, BuildAcceptorWater

    tensor = torch.DoubleTensor

    dist = tensor([2.65]).reshape(())
    angle = tensor([parse_angle("109.0 deg")]).reshape(())

    I = dict(  # noqa
        A=tensor((0.0, 3.0, 3.0)), B=tensor((0.0, 3.0, 4.0)), B0=tensor((1.0, 0.0, 0.0))
    )

    torsions = tensor([parse_angle(f"{a} deg") for a in (60.0, 300.0)])
    WI = torch.stack(
        [
            BuildAcceptorWater.apply(I["A"], I["B"], I["B0"], dist, angle, torsion)
            for torsion in torsions
        ]
    )

    J = dict(
        A=tensor((0.0, 0.0, 0.0)),
        B=tensor((0.0, 0.0, -1.0)),
        B0=tensor((1.0, 0.0, 0.0)),
    )

    torsions = tensor([parse_angle(f"{a} deg") for a in (120.0, 240.0)])
    WJ = torch.stack(
        [
            BuildAcceptorWater.apply(J["A"], J["B"], J["B0"], dist, angle, torsion)
            for torsion in torsions
        ]
    )

    lkbr_frac = LKBridgeFraction.apply(I["A"], J["A"], WI, WJ, dist)
    assert float(lkbr_frac) == pytest.approx(.025, abs=.001)

    gradcheck(
        lambda I, J, WI, WJ: LKBridgeFraction.apply(I, J, WI, WJ, dist),
        (
            I["A"].requires_grad_(True),
            J["A"].requires_grad_(True),
            WI.requires_grad_(True),
            WJ.requires_grad_(True),
        ),
        eps=1e-4,
    )


def lkball_score_and_gradcheck(
    ljlk_params, atype_params, I, J, WI, WJ, bonded_path_length, at_i, at_j
):
    from .compiled import LKBallScore, LKFraction, LKBridgeFraction

    aidx_i, aidx_j = ljlk_params.type_idx([at_i, at_j])

    op = LKBallScore(ljlk_params, atype_params)

    score = op.apply(I, J, WI, WJ, bonded_path_length, at_i, at_j)

    gradcheck(
        lambda WI, J: LKFraction.apply(
            WI, J, ljlk_params.type_params[aidx_j].lj_radius
        ),
        (WI.requires_grad_(True), J.requires_grad_(True)),
        eps=2e-3,
    )

    gradcheck(
        lambda I, J, WI, WJ: LKBridgeFraction.apply(
            I, J, WI, WJ, ljlk_params.global_params.lkb_water_dist
        ),
        (
            I.requires_grad_(True),
            J.requires_grad_(True),
            WI.requires_grad_(True),
            WJ.requires_grad_(True),
        ),
        eps=2e-3,
    )

    gradcheck(
        lambda I, J, WI, WJ: op.apply(I, J, WI, WJ, bonded_path_length, at_i, at_j),
        (
            I.requires_grad_(True),
            J.requires_grad_(False),
            WI.requires_grad_(False),
            WJ.requires_grad_(False),
        ),
        eps=2e-4,
    )

    return score


def test_lk_ball_donor_donor_spotcheck(ljlk_params, atype_params):

    from .compiled import BuildDonorWater

    dist = ljlk_params.global_params.lkb_water_dist

    tensor = torch.DoubleTensor

    coords = tensor(
        [
            [-6.007, 4.706, -0.074],
            [-6.747, 4.361, 0.549],
            [-5.791, 5.657, 0.240],
            [-6.305, 4.706, -1.040],
            [-10.018, 6.062, -2.221],
            [-9.160, 5.711, -2.665],
            [-9.745, 6.899, -1.697],
            [-10.429, 5.372, -1.610],
        ]
    )

    atom_types = ["Nlys", "Hpol", "Hpol", "Hpol", "Nlys", "Hpol", "Hpol", "Hpol"]

    coord_i = coords[0]
    at_i = atom_types[0]
    waters_i = torch.stack(
        [BuildDonorWater.apply(coords[0], H_c, dist) for H_c in coords[[1, 2, 3]]]
    )

    coord_j = coords[4]
    at_j = atom_types[4]
    waters_j = torch.stack(
        [BuildDonorWater.apply(coords[4], H_c, dist) for H_c in coords[[5, 6, 7]]]
    )

    bonded_path_length = tensor([5.0])[()]

    i_by_j = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        coord_i,
        coord_j,
        waters_i,
        waters_j,
        bonded_path_length,
        at_i,
        at_j,
    )
    j_by_i = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        coord_j,
        coord_i,
        waters_j,
        waters_i,
        bonded_path_length,
        at_i,
        at_j,
    )

    torch.testing.assert_allclose(
        i_by_j + j_by_i, tensor([0.3355, 0., 0.2649, 0.7896]), atol=1e-4, rtol=1e-4
    )


def test_lk_ball_sp2_nonpolar_spotcheck(ljlk_params, atype_params):

    from .compiled import BuildAcceptorWater

    tensor = torch.DoubleTensor
    coords = tensor(
        [
            [-5.282, -0.190, -0.858],
            [-6.520, 0.686, -0.931],
            [-6.652, 1.379, -1.961],
            [-6.932, 4.109, -5.683],
        ]
    )
    atom_types = ["CH2", "COO", "OOC", "CH3"]

    # sp2 acceptor coord
    sp2_at = atom_types[2]
    sp2_c = coords[2]
    sp2_waters = torch.stack(
        [
            BuildAcceptorWater.apply(
                coords[2],
                coords[1],
                coords[0],
                ljlk_params.global_params.lkb_water_dist,
                ljlk_params.global_params.lkb_water_angle_sp2,
                torsion,
            )
            for torsion in ljlk_params.global_params.lkb_water_tors_sp2
        ]
    )

    # nonpolar coord
    nonpolar_at = atom_types[3]
    nonpolar_c = coords[3]
    nonpolar_waters = torch.full_like(sp2_waters, math.nan)

    bonded_path_length = tensor([5.0])[()]

    i_by_j = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        sp2_c,
        nonpolar_c,
        sp2_waters,
        nonpolar_waters,
        bonded_path_length,
        sp2_at,
        nonpolar_at,
    )

    torch.testing.assert_allclose(i_by_j, tensor([0.14107985, 0.04765878, 0., 0.]))


def test_lk_ball_sp3_ring_spotcheck(ljlk_params, atype_params):
    from .compiled import BuildAcceptorWater, BuildDonorWater

    tensor = torch.DoubleTensor

    # test 3: ring acceptor--sp3 acceptor
    coords = tensor(
        [
            [-5.250, -1.595, -2.543],  # SER CB
            [-6.071, -0.619, -3.193],  # SER OG
            [-5.489, 0.060, -3.542],  # SER HG
            [-10.628, 2.294, -1.933],  # HIS CG
            [-9.991, 1.160, -1.435],  # HIS ND1
            [-10.715, 0.960, -0.319],  # HIS CE1
        ]
    )
    atom_types = ["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"]

    sp3_at = atom_types[1]
    sp3_c = coords[1]
    sp3_waters = torch.stack(
        [
            BuildAcceptorWater.apply(
                coords[1],
                coords[0],
                coords[2],
                ljlk_params.global_params.lkb_water_dist,
                ljlk_params.global_params.lkb_water_angle_sp3,
                torsion,
            )
            for torsion in ljlk_params.global_params.lkb_water_tors_sp3
        ]
        + [
            BuildDonorWater.apply(
                coords[1], coords[2], ljlk_params.global_params.lkb_water_dist
            )
        ]
    )

    ring_at = atom_types[4]
    ring_c = coords[4]
    ring_waters = torch.stack(
        # Hacky, expand ring water shape
        [
            BuildAcceptorWater.apply(
                coords[4],
                0.5 * (coords[3] + coords[5]),
                coords[5],
                ljlk_params.global_params.lkb_water_dist,
                ljlk_params.global_params.lkb_water_angle_ring,
                torsion,
            )
            for torsion in ljlk_params.global_params.lkb_water_tors_ring
        ]
        + [
            tensor([math.nan, math.nan, math.nan]),
            tensor([math.nan, math.nan, math.nan]),
        ]
    )

    assert ring_waters.shape == sp3_waters.shape
    bonded_path_length = tensor([5.0])[()]

    # Acceptor/Acceptor pair
    sp3_by_ring = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        sp3_c,
        ring_c,
        sp3_waters,
        ring_waters,
        bonded_path_length,
        sp3_at,
        ring_at,
    )
    ring_by_sp3 = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        ring_c,
        sp3_c,
        ring_waters,
        sp3_waters,
        bonded_path_length,
        ring_at,
        sp3_at,
    )

    scores = numpy.array([i + j for i, j in zip(sp3_by_ring, ring_by_sp3)])
    scores_ref = numpy.array([0.09018922, 0.0901892, 0.0441963, 0.4900393])
    assert scores == pytest.approx(scores_ref, abs=1e-4)

    sp3_by_nonpolar = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        sp3_c,
        coords[3],
        sp3_waters,
        torch.full_like(sp3_waters, math.nan),
        bonded_path_length,
        sp3_at,
        atom_types[3],
    )

    torch.testing.assert_allclose(
        sp3_by_nonpolar, tensor([0.00385956, 0.0001626, 0.0, 0.0]), atol=1e-4, rtol=1e-4
    )

    sp3_by_nonpolar = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        sp3_c,
        coords[5],
        sp3_waters,
        torch.full_like(sp3_waters, math.nan),
        bonded_path_length,
        sp3_at,
        atom_types[5],
    )

    torch.testing.assert_allclose(
        sp3_by_nonpolar, tensor([0.00369549, 0.0028072, 0.0, 0.0]), atol=1e-4, rtol=1e-4
    )

    ring_by_nonpolar = lkball_score_and_gradcheck(
        ljlk_params,
        atype_params,
        ring_c,
        coords[0],
        ring_waters,
        torch.full_like(ring_waters, math.nan),
        bonded_path_length,
        ring_at,
        atom_types[0],
    )
    torch.testing.assert_allclose(
        ring_by_nonpolar,
        tensor([0.01360676, 0.0135272, 0.0, 0.0]),
        atol=1e-4,
        rtol=1e-4,
    )
