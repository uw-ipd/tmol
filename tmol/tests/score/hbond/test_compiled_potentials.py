import pytest
from pytest import approx

import torch
from tmol.tests.autograd import gradcheck, VectorizedOp

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
    AcceptorType = compiled.AcceptorType

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

    atomD = [-0.337, 3.640, -1.365]
    atomH = [-0.045, 3.220, -0.496]
    atomA = [0.929, 2.820, 1.149]
    atomB = [1.369, 1.690, 1.360]
    atomB0 = [1.060, 0.538, 0.412]

    donwt = 1.45
    accwt = 1.19

    return dict(
        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        b0=atomB0,
        acceptor_type=AcceptorType.sp2,
        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeff=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeff=poly_cosBAH_off,
        cosBAH_range=poly_cosBAH_off_range,
        cosBAH_bound=poly_cosBAH_off_bounds,
        cosAHD_coeff=poly_AHD_1j,
        cosAHD_range=poly_AHD_1j_range,
        cosAHD_bound=poly_AHD_1j_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


@pytest.fixture
def sp3_params(compiled):
    AcceptorType = compiled.AcceptorType

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

    atomD = [-1.447, 4.942, -3.149]
    atomH = [-1.756, 4.013, -2.912]
    atomA = [-2.196, 2.211, -2.339]
    atomB = [-3.156, 2.109, -1.327]
    atomB0 = [-1.436, 1.709, -2.035]

    donwt = 1.45
    accwt = 1.15

    return dict(
        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        b0=atomB0,
        acceptor_type=AcceptorType.sp3,
        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeff=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeff=poly_cosBAH_6i,
        cosBAH_range=poly_cosBAH_6i_range,
        cosBAH_bound=poly_cosBAH_6i_bounds,
        cosAHD_coeff=poly_AHD_1i,
        cosAHD_range=poly_AHD_1i_range,
        cosAHD_bound=poly_AHD_1i_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


@pytest.fixture
def ring_params(compiled):
    AcceptorType = compiled.AcceptorType

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

    atomD = [-0.624, 5.526, -2.146]
    atomH = [-1.023, 4.664, -2.481]
    atomA = [-1.579, 2.834, -2.817]
    atomB = [-0.774, 1.927, -3.337]
    atomB0 = [-2.327, 2.261, -1.817]

    donwt = 1.45
    accwt = 1.13

    return dict(
        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        b0=atomB0,
        acceptor_type=AcceptorType.ring,
        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeff=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6,
        AHdist_range=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bound=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeff=poly_cosBAH_7,
        cosBAH_range=poly_cosBAH_7_range,
        cosBAH_bound=poly_cosBAH_7_bounds,
        cosAHD_coeff=poly_AHD_1i,
        cosAHD_range=poly_AHD_1i_range,
        cosAHD_bound=poly_AHD_1i_bounds,
        # Global score parameters
        **_hbond_global_params,
    )


def test_point_scores(compiled, sp2_params, sp3_params, ring_params):
    assert compiled.hbond_score(**sp2_params) == approx(-2.39, abs=0.01)
    assert compiled.hbond_score(**sp3_params) == approx(-2.00, abs=0.01)
    assert compiled.hbond_score(**ring_params) == approx(-2.17, abs=0.01)


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
                _t(params["AHdist_coeff"]),
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
                _t(params["a"]).requires_grad_(True),
                _t(params["h"]).requires_grad_(True),
                _t(params["d"]).requires_grad_(True),
                _t(params["cosAHD_coeff"]),
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
                _t(params["b"]).requires_grad_(True),
                _t(params["b0"]).requires_grad_(True),
                _t(params["a"]).requires_grad_(True),
                _t(params["h"]).requires_grad_(True),
                _t(params["acceptor_type"]).to(dtype=torch.int32),
                _t(params["cosBAH_coeff"]),
                _t(params["cosBAH_range"]),
                _t(params["cosBAH_bound"]),
                _t(params["hb_sp3_softmax_fade"]),
            ),
        )
