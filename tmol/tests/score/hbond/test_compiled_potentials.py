from pytest import approx


def test_sp2_single_hbond():
    from tmol.score.hbond.potentials.compiled import hbond_donor_sp2_score

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
        hb_sp2_range_span=1.6,
        hb_sp2_BAH180_rise=0.75,
        hb_sp2_outer_width=0.357,
    )

    # TODO Verify delta of .01 vs torch potential. Perhaps due to precision shift?
    assert energy == approx(-2.40, abs=.01)


def test_sp3_single_hbond():
    from tmol.score.hbond.potentials.compiled import hbond_donor_sp3_score

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
        hb_sp3_softmax_fade=2.5,
    )

    assert energy == approx(-2.00, abs=0.01)
