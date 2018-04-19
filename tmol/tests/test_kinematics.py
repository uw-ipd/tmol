import numpy
import numpy.testing

import pytest

from tmol.kinematics import (
    BOND, JUMP, kintree_node_dtype, backwardKin, forwardKin, resolveDerivs
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
def kintree():
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
    ## test interconversion
    HTs, dofs = backwardKin(kintree, coords)
    HTs, re_coords = forwardKin(kintree, dofs)
    numpy.testing.assert_allclose(coords, re_coords, atol=1e-9)


def test_perturb(kintree, coords):
    # test perturb
    HTs, dofs = backwardKin(kintree, coords)

    dofs[8, 3:6] = [0.02, 0.02, 0.02]
    dofs[13, 3:6] = [0.01, 0.01, 0.02]
    dofs[18, 3:6] = [0.01, 0.02, 0.01]
    (HTs, coords) = forwardKin(kintree, dofs)
    # TODO assertions


def test_derivs(kintree, coords):
    NATOMS, _ = coords.shape
    HTs, dofs = backwardKin(kintree, coords)

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

    # Compute numeric derivs
    dsc_dtors_numeric = numpy.zeros([NATOMS, 9])
    for i in numpy.arange(0, NATOMS):
        for j in numpy.arange(0, 6):
            dofs[i, j] += 0.00001
            (HTs, coordsAlt) = forwardKin(kintree, dofs)
            sc_p = score(kintree[3:], coordsAlt[3:, :])
            dofs[i, j] -= 0.00002
            (HTs, coordsAlt) = forwardKin(kintree, dofs)
            sc_m = score(kintree[3:], coordsAlt[3:, :])
            dofs[i, j] += 0.00001

            dsc_dtors_numeric[i, j] = (sc_p - sc_m) / 0.00002

    aderiv = dsc_dtors_analytic
    nderiv = dsc_dtors_numeric

    numpy.testing.assert_allclose(
        aderiv[jumps, :6], nderiv[jumps, :6], atol=1e-9
    )

    numpy.testing.assert_allclose(
        aderiv[bonds_after_bond, :3], nderiv[bonds_after_bond, :3], atol=1e-9
    )

    numpy.testing.assert_allclose(
        aderiv[bonds_after_jump, :3], nderiv[bonds_after_jump, :3], atol=1e-9
    )
