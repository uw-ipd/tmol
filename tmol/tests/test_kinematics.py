import numpy
import numpy.testing

import pytest

from tmol.kinematics import (
    DOFType, kintree_node_dtype, backwardKin, forwardKin, resolveDerivs
)


def score(coords):
    """Dummy scorefunction for a conformation."""
    assert coords.shape == (20, 3)
    dists = numpy.sqrt(
        numpy.square(coords[:, numpy.newaxis] - coords).sum(axis=2)
    )
    igraph = numpy.bitwise_and(
        numpy.triu(~numpy.eye(dists.shape[0], dtype=bool)), dists < 3.4
    ).nonzero()
    score = (3.4 - dists[igraph]) * (3.4 - dists[igraph])
    return numpy.sum(score)


def dscore(coords):
    """Dummy scorefunction derivs for a conformation."""
    assert coords.shape == (20, 3)
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
       [[0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
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
    ROOT = DOFType.root
    JUMP = DOFType.jump
    BOND = DOFType.bond
    NATOMS = 21

    # kinematics definition
    kintree = numpy.empty(NATOMS, dtype=kintree_node_dtype)
    kintree[0] = (0, ROOT, 0, 0, 0, 0)

    kintree[1] = (1, JUMP, 0, 2, 1, 3)
    kintree[2] = (1, BOND, 1, 2, 1, 3)
    kintree[3] = (1, BOND, 2, 3, 2, 1)
    kintree[4] = (1, BOND, 2, 4, 2, 1)
    kintree[5] = (1, BOND, 4, 5, 4, 2)

    kintree[6] = (2, JUMP, 1, 7, 6, 8)
    kintree[7] = (2, BOND, 6, 7, 6, 8)
    kintree[8] = (2, BOND, 7, 8, 7, 6)
    kintree[9] = (2, BOND, 7, 9, 7, 6)
    kintree[10] = (2, BOND, 9, 10, 9, 7)

    kintree[11] = (3, JUMP, 1, 12, 11, 13)
    kintree[12] = (3, BOND, 11, 12, 11, 13)
    kintree[13] = (3, BOND, 12, 13, 12, 11)
    kintree[14] = (3, BOND, 12, 14, 12, 11)
    kintree[15] = (3, BOND, 14, 15, 14, 12)

    kintree[16] = (4, JUMP, 1, 17, 16, 18)
    kintree[17] = (4, BOND, 16, 17, 16, 18)
    kintree[18] = (4, BOND, 17, 18, 17, 16)
    kintree[19] = (4, BOND, 17, 19, 17, 16)
    kintree[20] = (4, BOND, 19, 20, 19, 17)

    return kintree


@pytest.fixture
def coords():
    NATOMS = 21

    coords = numpy.empty([NATOMS, 3])
    coords[0, :] = [0.000, 0.000, 0.000]

    coords[1, :] = [2.000, 2.000, 2.000]
    coords[2, :] = [3.458, 2.000, 2.000]
    coords[3, :] = [3.988, 1.222, 0.804]
    coords[4, :] = [4.009, 3.420, 2.000]
    coords[5, :] = [3.383, 4.339, 1.471]

    coords[6, :] = [5.184, 3.594, 2.596]
    coords[7, :] = [5.821, 4.903, 2.666]
    coords[8, :] = [5.331, 5.667, 3.888]
    coords[9, :] = [7.339, 4.776, 2.690]
    coords[10, :] = [7.881, 3.789, 3.186]

    coords[11, :] = [7.601, 2.968, 5.061]
    coords[12, :] = [6.362, 2.242, 4.809]
    coords[13, :] = [6.431, 0.849, 5.419]
    coords[14, :] = [5.158, 3.003, 5.349]
    coords[15, :] = [5.265, 3.736, 6.333]

    coords[16, :] = [4.011, 2.824, 4.701]
    coords[17, :] = [2.785, 3.494, 5.115]
    coords[18, :] = [2.687, 4.869, 4.470]
    coords[19, :] = [1.559, 2.657, 4.776]
    coords[20, :] = [1.561, 1.900, 3.805]

    return coords


def test_score_smoketest(coords):
    score(coords[1:, :])


def test_interconversion(kintree, coords):
    bkin = backwardKin(kintree, coords)
    refold = forwardKin(kintree, bkin.dofs)
    numpy.testing.assert_allclose(coords, refold.coords, atol=1e-9)


def test_perturb(kintree, coords):
    dofs = backwardKin(kintree, coords).dofs

    pcoords = forwardKin(kintree, dofs).coords
    assert numpy.allclose(coords, pcoords)

    def coord_changed(a, b, atol=1e-3):
        return numpy.abs(a - b) > atol

    # Translate jump dof
    t_dofs = dofs.copy()
    t_dofs["jump"][6, :3] += [0.02] * 3
    pcoords = forwardKin(kintree, t_dofs).coords

    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6])
    assert numpy.all(coord_changed(pcoords[6:11], coords[6:11]))
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16])
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21])

    # Rotate jump dof "delta"
    rd_dofs = dofs.copy()
    numpy.testing.assert_allclose(rd_dofs["jump"][6, 3:6], [0, 0, 0])
    rd_dofs["jump"][6, 3:6] += [.1, .2, .3]
    pcoords = forwardKin(kintree, rd_dofs).coords
    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6])
    numpy.testing.assert_allclose(pcoords[6], coords[6])
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11], coords[7:11]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16])
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21])

    # Rotate jump dof
    r_dofs = dofs.copy()
    r_dofs["jump"][6, 6:9] += [.1, .2, .3]
    pcoords = forwardKin(kintree, r_dofs).coords
    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6])
    numpy.testing.assert_allclose(pcoords[6], coords[6])
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11], coords[7:11]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16])
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21])


def test_derivs(kintree, coords, expected_analytic_derivs):
    NATOMS, _ = coords.shape
    bkin = backwardKin(kintree, coords)
    HTs, dofs = bkin.hts, bkin.dofs

    bond_blocks = [
        numpy.arange(2, 6),
        numpy.arange(7, 11),
        numpy.arange(12, 16),
        numpy.arange(17, 21)
    ]
    jumps = numpy.array([b[-1] for b in bond_blocks[:-1]])
    bonds_after_bond = numpy.concatenate([b[1:] for b in bond_blocks])
    bonds_after_jump = numpy.array([b[0] for b in bond_blocks])

    # Compute analytic derivs
    dsc_dx = numpy.zeros([NATOMS, 3])
    dsc_dx[1:] = dscore(coords[1:, :])
    dsc_dtors_analytic = resolveDerivs(kintree, dofs, HTs, dsc_dx)

    # Verify against stored derivatives for regression
    numpy.testing.assert_allclose(
        dsc_dtors_analytic["raw"][1:],
        expected_analytic_derivs[1:],
    )

    # Compute numeric derivs
    dsc_dtors_numeric = numpy.zeros_like(dofs)
    for i in numpy.arange(0, NATOMS):
        if kintree[i]["doftype"] == DOFType.bond:
            ndof = 3
        elif kintree[i]["doftype"] == DOFType.jump:
            ndof = 6
        elif kintree[i]["doftype"] == DOFType.root:
            continue
        else:
            raise NotImplementedError

        for j in range(ndof):
            dofs["raw"][i, j] += 0.00001
            coordsAlt = forwardKin(kintree, dofs).coords
            sc_p = score(coordsAlt[1:, :])
            dofs["raw"][i, j] -= 0.00002
            coordsAlt = forwardKin(kintree, dofs).coords
            sc_m = score(coordsAlt[1:, :])
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
