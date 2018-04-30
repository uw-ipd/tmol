import numpy
import numpy.testing

import pytest

from tmol.kinematics import (
    DOFType, kintree_node_dtype, backwardKin, forwardKin, resolveDerivs
)


def score(kintree, coords):
    """Dummy scorefunction for a conformation."""
    dists = numpy.sqrt(
        numpy.square(coords[:, numpy.newaxis] - coords).sum(axis=2)
    )
    igraph = numpy.bitwise_and(
        numpy.triu(~numpy.eye(dists.shape[0], dtype=bool)), dists < 3.4
    ).nonzero()
    score = (3.4 - dists[igraph]) * (3.4 - dists[igraph])
    return numpy.sum(score)


def dscore(kintree, coords):
    """Dummy scorefunction derivs for a conformation."""
    natoms = coords.shape[0]
    dxs = coords[:, numpy.newaxis] - coords
    dists = numpy.sqrt(numpy.square(dxs).sum(axis=2))
    igraph = numpy.bitwise_and(
        numpy.triu(~numpy.eye(dists.shape[0], dtype=bool)), dists < 3.4
    ).nonzero()

    dEdxs = numpy.zeros([natoms, natoms, 3])
    dEdxs[igraph[0], igraph[1], :] = -2 * (
        3.4 - dists[igraph].reshape(-1, 1)
    ) * dxs[igraph] / dists[igraph].reshape(-1, 1)

    dEdx = numpy.zeros([natoms, 3])
    dEdx = dEdxs.sum(axis=1) - dEdxs.sum(axis=0)

    return dEdx


@pytest.fixture
def expected_analytic_derivs():
    return numpy.array(
       [[3.69397049e-01, -4.06528744e+01,  1.20658609e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [2.66453526e-15,  1.77635684e-15, -8.88178420e-16,
         5.32907052e-15,  3.55271368e-15,  4.44089210e-15,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [8.61394482e-01, -1.06522516e+01,  1.13718660e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-7.30414633e+00,  3.40681357e-01,  1.70214350e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-5.80415867e+00, -1.04967293e+01,  9.66972247e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-9.52873353e+00, -6.06030702e-02,  7.68770907e-01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-1.25079554e+01,  1.57258359e+00, -8.89832569e+00,
         5.52111345e+00,  7.40141077e+00, -9.92561863e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-1.47396223e+01,  8.50512385e+00, -8.72247305e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-6.81301657e+00,  4.52266852e+00,  1.70112178e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-1.08827823e+01,  7.57222441e+00, -1.04235948e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-4.52952181e+00,  5.88968488e+00, -1.93873880e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-9.66405135e+00,  7.51131337e+00,  9.49787678e+00,
         2.14775674e-01,  1.47298659e+01, -1.71810938e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [1.16243389e+00, -1.05854011e+01,  1.15033592e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-7.30134090e+00,  3.43791230e-01,  1.69990508e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-5.85401170e+00, -1.05263343e+01,  9.80345414e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-9.57540886e+00, -4.76323730e-02,  7.74393479e-01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [1.45955969e+01,  3.92793100e+00,  3.79171849e+00,
         2.46056151e+00,  1.02226395e+01, -9.80368607e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-1.51385559e+01,  6.84468376e+00, -9.32753028e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-6.91878652e+00,  3.87588970e+00,  1.62693321e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-1.05711446e+01,  9.24780005e+00, -1.09544635e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-3.75430706e+00,  6.15307006e+00, -1.79443222e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    ) # yapf: disable


@pytest.fixture
def kintree():
    BOND = DOFType.bond
    JUMP = DOFType.jump
    NATOMS = 23

    # kinematics definition
    kintree = numpy.empty(NATOMS, dtype=kintree_node_dtype)
    kintree[0] = ("ORIG", 0, BOND, 0, (1, 0, 2))
    kintree[1] = (" X  ", 0, BOND, 0, (1, 0, 2))
    kintree[2] = (" Y  ", 0, BOND, 0, (2, 0, 1))

    kintree[3] = (" N  ", 1, JUMP, 0, (4, 3, 5))
    kintree[4] = (" CA ", 1, BOND, 3, (4, 3, 5))
    kintree[5] = (" CB ", 1, BOND, 4, (5, 4, 3))
    kintree[6] = (" C  ", 1, BOND, 4, (6, 4, 3))
    kintree[7] = (" O  ", 1, BOND, 6, (7, 6, 4))

    kintree[8] = (" N  ", 2, JUMP, 3, (9, 8, 10))
    kintree[9] = (" CA ", 2, BOND, 8, (9, 8, 10))
    kintree[10] = (" CB ", 2, BOND, 9, (10, 9, 8))
    kintree[11] = (" C  ", 2, BOND, 9, (11, 9, 8))
    kintree[12] = (" O  ", 2, BOND, 11, (12, 11, 9))

    kintree[13] = (" N  ", 3, JUMP, 3, (14, 13, 15))
    kintree[14] = (" CA ", 3, BOND, 13, (14, 13, 15))
    kintree[15] = (" CB ", 3, BOND, 14, (15, 14, 13))
    kintree[16] = (" C  ", 3, BOND, 14, (16, 14, 13))
    kintree[17] = (" O  ", 3, BOND, 16, (17, 16, 14))

    kintree[18] = (" N  ", 4, JUMP, 3, (19, 18, 20))
    kintree[19] = (" CA ", 4, BOND, 18, (19, 18, 20))
    kintree[20] = (" CB ", 4, BOND, 19, (20, 19, 18))
    kintree[21] = (" C  ", 4, BOND, 19, (21, 19, 18))
    kintree[22] = (" O  ", 4, BOND, 21, (22, 21, 19))

    return kintree


@pytest.fixture
def coords():
    NATOMS = 23

    coords = numpy.empty([NATOMS, 3])
    coords[0, :] = [0.000, 0.000, 0.000]
    coords[1, :] = [1.000, 0.000, 0.000]
    coords[2, :] = [0.000, 1.000, 0.000]

    coords[3, :] = [2.000, 2.000, 2.000]
    coords[4, :] = [3.458, 2.000, 2.000]
    coords[5, :] = [3.988, 1.222, 0.804]
    coords[6, :] = [4.009, 3.420, 2.000]
    coords[7, :] = [3.383, 4.339, 1.471]

    coords[8, :] = [5.184, 3.594, 2.596]
    coords[9, :] = [5.821, 4.903, 2.666]
    coords[10, :] = [5.331, 5.667, 3.888]
    coords[11, :] = [7.339, 4.776, 2.690]
    coords[12, :] = [7.881, 3.789, 3.186]

    coords[13, :] = [7.601, 2.968, 5.061]
    coords[14, :] = [6.362, 2.242, 4.809]
    coords[15, :] = [6.431, 0.849, 5.419]
    coords[16, :] = [5.158, 3.003, 5.349]
    coords[17, :] = [5.265, 3.736, 6.333]

    coords[18, :] = [4.011, 2.824, 4.701]
    coords[19, :] = [2.785, 3.494, 5.115]
    coords[20, :] = [2.687, 4.869, 4.470]
    coords[21, :] = [1.559, 2.657, 4.776]
    coords[22, :] = [1.561, 1.900, 3.805]

    return coords


def test_score_smoketest(kintree, coords):
    score(kintree[3:], coords[3:, :])


def test_interconversion(kintree, coords):
    dofs = backwardKin(kintree, coords).dofs
    HTs, re_coords = forwardKin(kintree, dofs)
    numpy.testing.assert_allclose(coords, re_coords, atol=1e-9)


def test_perturb(kintree, coords):
    dofs = backwardKin(kintree, coords).dofs

    (HTs, pcoords) = forwardKin(kintree, dofs)
    assert numpy.allclose(coords, pcoords)

    def coord_changed(a, b, atol=1e-3):
        return numpy.abs(a - b) > atol

    # Translate jump dof
    t_dofs = dofs.copy()
    t_dofs["jump"][8, :3] += [0.02] * 3
    (HTs, pcoords) = forwardKin(kintree, t_dofs)

    numpy.testing.assert_allclose(pcoords[3:8], coords[3:8])
    assert numpy.all(coord_changed(pcoords[8:13], coords[8:13]))
    numpy.testing.assert_allclose(pcoords[13:18], coords[13:18])
    numpy.testing.assert_allclose(pcoords[18:23], coords[18:23])

    # Rotate jump dof "delta"
    rd_dofs = dofs.copy()
    numpy.testing.assert_allclose(rd_dofs["jump"][8, 3:6], [0, 0, 0])
    rd_dofs["jump"][8, 3:6] += [.1, .2, .3]
    (HTs, pcoords) = forwardKin(kintree, rd_dofs)
    numpy.testing.assert_allclose(pcoords[3:8], coords[3:8])
    numpy.testing.assert_allclose(pcoords[8], coords[8])
    assert numpy.all(
        numpy.any(coord_changed(pcoords[9:13], coords[9:13]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[13:18], coords[13:18])
    numpy.testing.assert_allclose(pcoords[18:23], coords[18:23])

    # Rotate jump dof
    r_dofs = dofs.copy()
    r_dofs["jump"][8, 6:9] += [.1, .2, .3]
    (HTs, pcoords) = forwardKin(kintree, r_dofs)
    numpy.testing.assert_allclose(pcoords[3:8], coords[3:8])
    numpy.testing.assert_allclose(pcoords[8], coords[8])
    assert numpy.all(
        numpy.any(coord_changed(pcoords[9:13], coords[9:13]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[13:18], coords[13:18])
    numpy.testing.assert_allclose(pcoords[18:23], coords[18:23])


def test_derivs(kintree, coords, expected_analytic_derivs):
    NATOMS, _ = coords.shape
    bkin = backwardKin(kintree, coords)
    HTs, dofs = bkin.hts, bkin.dofs

    bond_blocks = [
        numpy.arange(4, 8),
        numpy.arange(9, 13),
        numpy.arange(14, 18),
        numpy.arange(19, 23)
    ]
    jumps = numpy.array([b[-1] for b in bond_blocks[:-1]])
    bonds_after_bond = numpy.concatenate([b[1:] for b in bond_blocks])
    bonds_after_jump = numpy.array([b[0] for b in bond_blocks])

    # Compute analytic derivs
    dsc_dx = numpy.zeros([NATOMS, 3])
    dsc_dx[3:, :] = dscore(kintree[3:], coords[3:, :])
    dsc_dtors_analytic = resolveDerivs(kintree, dofs, HTs, dsc_dx)

    # Verify against stored derivatives for regression
    numpy.testing.assert_allclose(
        dsc_dtors_analytic["raw"][3:],
        expected_analytic_derivs[3:],
    )

    # Compute numeric derivs
    dsc_dtors_numeric = numpy.zeros_like(dofs)
    for i in numpy.arange(0, NATOMS):
        if kintree[i]["doftype"] == DOFType.bond:
            ndof = 3
        elif kintree[i]["doftype"] == DOFType.jump:
            ndof = 6
        else:
            raise NotImplementedError

        for j in range(ndof):
            dofs["raw"][i, j] += 0.00001
            (HTs, coordsAlt) = forwardKin(kintree, dofs)
            sc_p = score(kintree[3:], coordsAlt[3:, :])
            dofs["raw"][i, j] -= 0.00002
            (HTs, coordsAlt) = forwardKin(kintree, dofs)
            sc_m = score(kintree[3:], coordsAlt[3:, :])
            dofs["raw"][i, j] += 0.00001

            dsc_dtors_numeric["raw"][i, j] = (sc_p - sc_m) / 0.00002

    # Verify numeric/analytic derivatives
    aderiv = dsc_dtors_analytic
    nderiv = dsc_dtors_numeric

    numpy.testing.assert_allclose(
        aderiv["jump"][jumps], nderiv["jump"][jumps], atol=1e-9
    )

    numpy.testing.assert_allclose(
        aderiv["bond"][bonds_after_bond],
        nderiv["bond"][bonds_after_bond],
        atol=1e-9
    )

    numpy.testing.assert_allclose(
        aderiv["bond"][bonds_after_jump],
        nderiv["bond"][bonds_after_jump],
        atol=1e-9
    )
