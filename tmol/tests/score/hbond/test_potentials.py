import pytest

import torch

from tmol.score.hbond.potentials import (
    hbond_donor_sp2_score,
    hbond_donor_sp3_score,
    hbond_donor_ring_score,
)


def test_sp2_single_hbond():
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -0.5307601, 6.47949946, -22.39522814, -55.14303544, 708.30945242,
        -2619.49318162, 5227.8805795, -6043.31211632, 3806.04676175,
        -1007.66024144
    ]])
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.38403812683, 2.9981039433
    ]])
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_off = torch.tensor([[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]])
    poly_cosBAH_off_range = torch.tensor([[-1234.0, 1.1]])
    poly_cosBAH_off_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1j = torch.tensor([[
        0.0, 0.47683259, -9.54524724, 83.62557693, -420.55867774,
        1337.19354878, -2786.26265686, 3803.178227, -3278.62879901,
        1619.04116204, -347.50157909
    ]])

    poly_AHD_1j_range = torch.tensor([[1.1435646388, 3.1416]])
    poly_AHD_1j_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-0.337, 3.640, -1.365]])
    atomH = torch.tensor([[-0.045, 3.220, -0.496]])
    atomA = torch.tensor([[0.929, 2.820, 1.149]])
    atomB = torch.tensor([[1.369, 1.690, 1.360]])
    atomB0 = torch.tensor([[1.060, 0.538, 0.412]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.19]])

    energy = hbond_donor_sp2_score(
        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        b0=atomB0,

        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeffs=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6,
        AHdist_ranges=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bounds=hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_off,
        cosBAH_ranges=poly_cosBAH_off_range,
        cosBAH_bounds=poly_cosBAH_off_bounds,
        cosAHD_coeffs=poly_AHD_1j,
        cosAHD_ranges=poly_AHD_1j_range,
        cosAHD_bounds=poly_AHD_1j_bounds,

        # Global score parameters
        hb_sp2_range_span=1.6,
        hb_sp2_BAH180_rise=0.75,
        hb_sp2_outer_width=0.357,
    )

    assert (float(energy.data[0]) == pytest.approx(-2.41, 0.01))


def test_sp3_single_hbond():
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -1.32847415, 22.67528654, -172.53450064, 770.79034865,
        -2233.48829652, 4354.38807288, -5697.35144236, 4803.38686157,
        -2361.48028857, 518.28202382
    ]])

    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.38565621563, 2.74160605537
    ]])
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_6i = torch.tensor([[
        0.0, -0.82209300, -3.75364636, 46.88852157, -129.54405640,
        146.69151428, -67.60598792, 2.91683129, 9.26673173, -3.84488178,
        0.05706659
    ]])
    poly_cosBAH_6i_range = torch.tensor([[-0.0193738506669, 1.1]])
    poly_cosBAH_6i_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1i = torch.tensor([[
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
    ]])
    poly_AHD_1i_range = torch.tensor([[1.59914724347, 3.1416]])
    poly_AHD_1i_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-1.447, 4.942, -3.149]])
    atomH = torch.tensor([[-1.756, 4.013, -2.912]])
    atomA = torch.tensor([[-2.196, 2.211, -2.339]])
    atomB = torch.tensor([[-3.156, 2.109, -1.327]])
    atomB0 = torch.tensor([[-1.436, 1.709, -2.035]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.15]])

    energy = hbond_donor_sp3_score(

        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        b0=atomB0,

        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeffs=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6,
        AHdist_ranges=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bounds=hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_6i,
        cosBAH_ranges=poly_cosBAH_6i_range,
        cosBAH_bounds=poly_cosBAH_6i_bounds,
        cosAHD_coeffs=poly_AHD_1i,
        cosAHD_ranges=poly_AHD_1i_range,
        cosAHD_bounds=poly_AHD_1i_bounds,

        # Global score parameters
        hb_sp3_softmax_fade=2.5,
    )

    assert (float(energy.data[0]) == pytest.approx(-2.00, 0.01))


def test_ring_single_hbond():

    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -1.68095217, 21.31894078, -107.72203494, 251.81021758,
        -134.07465831, -707.64527046, 1894.62827430, -2156.85951846,
        1216.83585872, -275.48078944
    ]])
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.01629363411, 2.58523052904
    ]])
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_7 = torch.tensor([[
        0.0, 0.0, -27.942923450028001, 136.039920253367995,
        -268.069590567470016, 275.400462507918974, -153.502076215948989,
        39.741591385461000, 0.693861510121000, -3.885952320499000,
        1.024765090788892
    ]])
    poly_cosBAH_7_range = torch.tensor([[-0.0193738506669, 1.1]])
    poly_cosBAH_7_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1i = torch.tensor([[
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
    ]])
    poly_AHD_1i_range = torch.tensor([[1.59914724347, 3.1416]])
    poly_AHD_1i_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-0.624, 5.526, -2.146]])
    atomH = torch.tensor([[-1.023, 4.664, -2.481]])
    atomA = torch.tensor([[-1.579, 2.834, -2.817]])
    atomB = torch.tensor([[-0.774, 1.927, -3.337]])
    atomB0 = torch.tensor([[-2.327, 2.261, -1.817]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.13]])

    energy = hbond_donor_ring_score(
        # Input coordinates
        d=atomD,
        h=atomH,
        a=atomA,
        b=atomB,
        bp=atomB0,

        # type pair parameters
        glob_accwt=accwt,
        glob_donwt=donwt,
        AHdist_coeffs=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6,
        AHdist_ranges=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range,
        AHdist_bounds=hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds,
        cosBAH_coeffs=poly_cosBAH_7,
        cosBAH_ranges=poly_cosBAH_7_range,
        cosBAH_bounds=poly_cosBAH_7_bounds,
        cosAHD_coeffs=poly_AHD_1i,
        cosAHD_ranges=poly_AHD_1i_range,
        cosAHD_bounds=poly_AHD_1i_bounds,
    )

    assert (float(energy.data[0]) == pytest.approx(-2.17, 0.01))
