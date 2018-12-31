import pytest
from pytest import approx
from toolz import valmap

import numpy
import torch
from tmol.tests.autograd import gradcheck, VectorizedOp
from tmol.utility.args import _signature

from tmol.score.hbond.params import AcceptorClass


_hbond_global_params = dict(
    hb_sp2_range_span=1.6,
    hb_sp2_BAH180_rise=0.75,
    hb_sp2_outer_width=0.357,
    hb_sp3_softmax_fade=2.5,
)


@pytest.fixture
def compiled(scope="session"):
    """Move compilation to test fixture to report compilation errors as test failure."""
    import tmol.score.hbond.potentials.compiled

    return tmol.score.hbond.potentials.compiled


@pytest.fixture
def sp2_params(compiled):
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6 = [
        0.0,
        -0.5307601,
        6.47949946,
        -22.39522814,
        -55.14303544,
        708.30945242,
        -2619.49318162,
        5227.8805795,
        -6043.31211632,
        3806.04676175,
        -1007.66024144,
    ]

    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range = [1.38403812683, 2.9981039433]
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds = [1.1, 1.1]

    poly_cosBAH_off = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    poly_cosBAH_off_range = [-1234.0, 1.1]
    poly_cosBAH_off_bounds = [1.1, 1.1]

    poly_AHD_1j = [
        0.0,
        0.47683259,
        -9.54524724,
        83.62557693,
        -420.55867774,
        1337.19354878,
        -2786.26265686,
        3803.178227,
        -3278.62879901,
        1619.04116204,
        -347.50157909,
    ]

    poly_AHD_1j_range = [1.1435646388, 3.1416]
    poly_AHD_1j_bounds = [1.1, 1.1]

    return dict(
        # Input coordinates
        D=[-0.337, 3.640, -1.365],
        H=[-0.045, 3.220, -0.496],
        A=[0.929, 2.820, 1.149],
        B=[1.369, 1.690, 1.360],
        B0=[1.060, 0.538, 0.412],
        acceptor_class=AcceptorClass.sp2,
        # type pair parameters
        donor_weight=1.45,
        acceptor_weight=1.19,
        AHdist_coeffs=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_off,
        cosBAH_range=poly_cosBAH_off_range,
        cosBAH_bound=poly_cosBAH_off_bounds,
        cosAHD_coeffs=poly_AHD_1j,
        cosAHD_range=poly_AHD_1j_range,
        cosAHD_bound=poly_AHD_1j_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


@pytest.fixture
def sp3_params(compiled):
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6 = [
        0.0,
        -1.32847415,
        22.67528654,
        -172.53450064,
        770.79034865,
        -2233.48829652,
        4354.38807288,
        -5697.35144236,
        4803.38686157,
        -2361.48028857,
        518.28202382,
    ]

    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range = [1.38565621563, 2.74160605537]
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds = [1.1, 1.1]

    poly_cosBAH_6i = [
        0.0,
        -0.82209300,
        -3.75364636,
        46.88852157,
        -129.54405640,
        146.69151428,
        -67.60598792,
        2.91683129,
        9.26673173,
        -3.84488178,
        0.05706659,
    ]
    poly_cosBAH_6i_range = [-0.0193738506669, 1.1]
    poly_cosBAH_6i_bounds = [1.1, 1.1]

    poly_AHD_1i = [
        0.0,
        -0.18888801,
        3.48241679,
        -25.65508662,
        89.57085435,
        -95.91708218,
        -367.93452341,
        1589.69047020,
        -2662.35821350,
        2184.40194483,
        -723.28383545,
    ]
    poly_AHD_1i_range = [1.59914724347, 3.1416]
    poly_AHD_1i_bounds = [1.1, 1.1]

    return dict(
        # Input coordinates
        D=[-1.447, 4.942, -3.149],
        H=[-1.756, 4.013, -2.912],
        A=[-2.196, 2.211, -2.339],
        B=[-3.156, 2.109, -1.327],
        B0=[-1.436, 1.709, -2.035],
        acceptor_class=AcceptorClass.sp3,
        # type pair parameters
        donor_weight=1.45,
        acceptor_weight=1.15,
        AHdist_coeffs=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_6i,
        cosBAH_range=poly_cosBAH_6i_range,
        cosBAH_bound=poly_cosBAH_6i_bounds,
        cosAHD_coeffs=poly_AHD_1i,
        cosAHD_range=poly_AHD_1i_range,
        cosAHD_bound=poly_AHD_1i_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


@pytest.fixture
def ring_params(compiled):
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6 = [
        0.0,
        -1.68095217,
        21.31894078,
        -107.72203494,
        251.81021758,
        -134.07465831,
        -707.64527046,
        1894.62827430,
        -2156.85951846,
        1216.83585872,
        -275.48078944,
    ]
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range = [1.01629363411, 2.58523052904]
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds = [1.1, 1.1]

    poly_cosBAH_7 = [
        0.0,
        0.0,
        -27.942923450028001,
        136.039920253367995,
        -268.069590567470016,
        275.400462507918974,
        -153.502076215948989,
        39.741591385461000,
        0.693861510121000,
        -3.885952320499000,
        1.024765090788892,
    ]
    poly_cosBAH_7_range = [-0.0193738506669, 1.1]
    poly_cosBAH_7_bounds = [1.1, 1.1]

    poly_AHD_1i = [
        0.0,
        -0.18888801,
        3.48241679,
        -25.65508662,
        89.57085435,
        -95.91708218,
        -367.93452341,
        1589.69047020,
        -2662.35821350,
        2184.40194483,
        -723.28383545,
    ]
    poly_AHD_1i_range = [1.59914724347, 3.1416]
    poly_AHD_1i_bounds = [1.1, 1.1]

    return dict(
        # Input coordinates
        D=[-0.624, 5.526, -2.146],
        H=[-1.023, 4.664, -2.481],
        A=[-1.579, 2.834, -2.817],
        B=[-0.774, 1.927, -3.337],
        B0=[-2.327, 2.261, -1.817],
        acceptor_class=AcceptorClass.ring,
        # type pair parameters
        donor_weight=1.45,
        acceptor_weight=1.13,
        AHdist_coeffs=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_7,
        cosBAH_range=poly_cosBAH_7_range,
        cosBAH_bound=poly_cosBAH_7_bounds,
        cosAHD_coeffs=poly_AHD_1i,
        cosAHD_range=poly_AHD_1i_range,
        cosAHD_bound=poly_AHD_1i_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


def test_hbond_point_scores(compiled, sp2_params, sp3_params, ring_params):
    assert compiled.hbond_score_V_dV(**sp2_params)[0] == approx(-2.40, abs=0.01)
    assert compiled.hbond_score_V_dV(**sp3_params)[0] == approx(-2.00, abs=0.01)
    assert compiled.hbond_score_V_dV(**ring_params)[0] == approx(-2.17, abs=0.01)


def test_hbond_point_scores_gradcheck(compiled, sp2_params, sp3_params, ring_params):
    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    def targs(params):
        args = (
            _signature(compiled.hbond_score_V_dV).bind(**valmap(_t, params)).arguments
        )

        args["D"] = args["D"].requires_grad_(True)
        args["H"] = args["H"].requires_grad_(True)
        args["A"] = args["A"].requires_grad_(True)
        args["B"] = args["B"].requires_grad_(True)
        args["B0"] = args["B0"].requires_grad_(True)
        args["acceptor_class"] = args["acceptor_class"].to(dtype=torch.int32)
        return tuple(args.values())

    op = VectorizedOp(compiled.hbond_score_V_dV)

    assert float(op(*targs(sp2_params))) == approx(-2.40, abs=0.01)
    gradcheck(op, targs(sp2_params))

    assert float(op(*targs(sp3_params))) == approx(-2.00, abs=0.01)
    gradcheck(op, targs(sp3_params))

    assert float(op(*targs(ring_params))) == approx(-2.17, abs=0.01)
    gradcheck(op, targs(ring_params))


def test_AH_dist_gradcheck(compiled, sp2_params, sp3_params, ring_params):
    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    for params in (sp2_params, sp3_params, ring_params):

        A = _t([[0.1, 0.1, v] for v in torch.arange(0, 3, 0.1)])
        H = _t([0, 0, 0])

        gradcheck(
            VectorizedOp(compiled.AH_dist_V_dV),
            (
                A.requires_grad_(True),
                H.requires_grad_(True),
                _t(params["AHdist_coeffs"]),
                _t(params["AHdist_range"]),
                _t(params["AHdist_bound"]),
            ),
        )


def test_AHD_angle_gradcheck(compiled, sp2_params, sp3_params, ring_params):
    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    for params in (sp2_params, sp3_params, ring_params):
        gradcheck(
            VectorizedOp(compiled.AHD_angle_V_dV),
            (
                _t(params["A"]).requires_grad_(True),
                _t(params["H"]).requires_grad_(True),
                _t(params["D"]).requires_grad_(True),
                _t(params["cosAHD_coeffs"]),
                _t(params["cosAHD_range"]),
                _t(params["cosAHD_bound"]),
            ),
        )


def test_BAH_angle_gradcheck(compiled, sp2_params, sp3_params, ring_params):
    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    for params in (sp2_params, sp3_params, ring_params):
        gradcheck(
            VectorizedOp(compiled.BAH_angle_V_dV),
            (
                _t(params["B"]).requires_grad_(True),
                _t(params["B0"]).requires_grad_(True),
                _t(params["A"]).requires_grad_(True),
                _t(params["H"]).requires_grad_(True),
                _t(params["acceptor_class"]).to(dtype=torch.int32),
                _t(params["cosBAH_coeffs"]),
                _t(params["cosBAH_range"]),
                _t(params["cosBAH_bound"]),
                _t(params["hb_sp3_softmax_fade"]),
            ),
        )


def test_sp2_chi_energy_gradcheck(compiled, sp2_params):
    from tmol.score.common.geom import dihedral_angle_V

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    chi = dihedral_angle_V(
        sp2_params["B0"], sp2_params["B"], sp2_params["A"], sp2_params["H"]
    )

    for ang in list(numpy.linspace(.1, numpy.pi, 16, endpoint=False)):
        params = (
            _t(ang).requires_grad_(True),
            _t(chi).requires_grad_(True),
            _t(sp2_params["hb_sp2_BAH180_rise"]),
            _t(sp2_params["hb_sp2_range_span"]),
            _t(sp2_params["hb_sp2_outer_width"]),
        )
        gradcheck(VectorizedOp(compiled.sp2chi_energy_V_dV), params, eps=1e-2)
