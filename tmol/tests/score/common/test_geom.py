import pytest
from pytest import approx

import torch

from numpy import array, cos, sin, linspace, pi, nan
from scipy.stats import special_ortho_group

from tmol.tests.autograd import gradcheck, VectorizedOp


def angle_vecs(theta):
    return [[10.0, 0.0, 0.0], [2 * cos(theta), 2 * sin(theta), 0.0]]


@pytest.fixture(scope="session")
def geom():
    import tmol.score.common.geom

    return tmol.score.common.geom


# TODO Test cases for colinear and identical points.
def test_interior_angle_values(geom):

    thetas = linspace(0, pi, 250, endpoint=False)
    rot = special_ortho_group.rvs(3)

    for theta in thetas:
        v1, v2 = angle_vecs(theta)

        assert geom.interior_angle_V(v1, v2) == approx(theta)
        assert geom.interior_angle_V_dV(v1, v2)[0] == approx(theta)

        assert geom.interior_angle_V(v1 @ rot, v2 @ rot) == approx(theta)
        assert geom.interior_angle_V_dV(v1 @ rot, v2 @ rot)[0] == approx(theta)


def test_interior_angle_gradcheck(geom):

    thetas = linspace(1e-5, pi, 100, endpoint=False)
    v1, v2 = array(list(map(angle_vecs, thetas))).swapaxes(0, 1)

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(geom.interior_angle_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
    )


def test_cos_interior_angle_values(geom):
    thetas = linspace(0, pi, 250, endpoint=False)
    rot = special_ortho_group.rvs(3)

    for theta in thetas:
        v1, v2 = angle_vecs(theta)

        assert geom.cos_interior_angle_V(v1, v2) == approx(cos(theta))
        assert geom.cos_interior_angle_V_dV(v1, v2)[0] == approx(cos(theta))

        assert geom.cos_interior_angle_V(v1 @ rot, v2 @ rot) == approx(cos(theta))
        assert geom.cos_interior_angle_V_dV(v1 @ rot, v2 @ rot)[0] == approx(cos(theta))


def test_cos_interior_angle_gradcheck(geom):

    thetas = linspace(1e-5, pi, 100, endpoint=False)
    v1, v2 = array(list(map(angle_vecs, thetas))).swapaxes(0, 1)

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(geom.cos_interior_angle_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
    )


def dihedral_points(chi):
    v1, v2 = angle_vecs(chi)

    return [[v1[0], v1[1], 0], [0, 0, .5], [0, 0, 1.75], [v2[0], v2[1], 2.25]]


def test_dihedral_angle_values(geom):
    chis = linspace(-pi, pi, 361, endpoint=False)
    rot = special_ortho_group.rvs(3)

    for chi in chis:
        I, J, K, L = dihedral_points(chi)

        assert geom.dihedral_angle_V(I, J, K, L) == approx(chi)
        assert geom.dihedral_angle_V(I @ rot, J @ rot, K @ rot, L @ rot) == approx(chi)


def test_coord_dihedrals(geom):
    from tmol.utility.units import parse_angle

    coords = array(
        [
            [24.969, 13.428, 30.692],  # N
            [24.044, 12.661, 29.808],  # CA
            [22.785, 13.482, 29.543],  # C
            [21.951, 13.670, 30.431],  # O
            [23.672, 11.328, 30.466],  # CB
            [22.881, 10.326, 29.620],  # CG
            [23.691, 9.935, 28.389],  # CD1
            [22.557, 9.096, 30.459],  # CD2
            [nan, nan, nan],
        ]
    )

    dihedral_atoms = array(
        [[0, 1, 2, 3], [0, 1, 4, 5], [1, 4, 5, 6], [1, 4, 5, 7], [8, 0, 1, 3]]
    )

    dihedrals = array(
        list(
            map(
                parse_angle,
                [
                    "-71.21515 deg",
                    "-171.94319 deg",
                    "60.82226 deg",
                    "-177.63641 deg",
                    nan,
                ],
            )
        )
    )

    vals = geom.dihedral_angle_V(
        coords[dihedral_atoms.T[0]],
        coords[dihedral_atoms.T[1]],
        coords[dihedral_atoms.T[2]],
        coords[dihedral_atoms.T[3]],
    )

    assert vals == approx(dihedrals, nan_ok=True)
