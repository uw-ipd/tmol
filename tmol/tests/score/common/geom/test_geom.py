import attr

import pytest
from pytest import approx

import torch

from numpy import array, cos, sin, linspace, pi, nan
from scipy.stats import special_ortho_group

from tmol.tests.autograd import gradcheck, VectorizedOp


@pytest.fixture(scope="session")
def geom():
    import tmol.tests.score.common.geom.geom as geom

    return geom


def dist_vecs(dist):
    return [[-dist / 2, 0.0, 0.0], [dist / 2, 0.0, 0.0]]


def test_distance_values(geom):
    dists = linspace(0, 10, 100, endpoint=True)
    rot = special_ortho_group.rvs(3)
    for dist in dists:
        v1, v2 = dist_vecs(dist)

        assert geom.distance_V(v1, v2) == approx(dist)
        assert geom.distance_V_dV(v1, v2)[0] == approx(dist)

        assert geom.distance_V(v1 @ rot, v2 @ rot) == approx(dist)
        assert geom.distance_V_dV(v1 @ rot, v2 @ rot)[0] == approx(dist)


def test_distance_gradcheck(geom):

    dists = linspace(0, 10, 100, endpoint=True)
    v1, v2 = array(list(map(dist_vecs, dists))).swapaxes(0, 1)

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(geom.distance_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
        eps=1e-3,
    )


def angle_vecs(theta):
    return [[10.0, 0.0, 0.0], [2 * cos(theta), 2 * sin(theta), 0.0]]


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
    chis = linspace(-pi, pi, 361, endpoint=False)[1:]
    rot = special_ortho_group.rvs(3)

    for chi in chis:
        I, J, K, L = dihedral_points(chi)

        assert geom.dihedral_angle_V(I, J, K, L) == approx(chi)
        # TODO flaky nan-values on some random rotations w/ chi=-pi
        assert geom.dihedral_angle_V(I @ rot, J @ rot, K @ rot, L @ rot) == approx(chi)


def test_dihedral_angle_values_gradcheck(geom):
    chis = linspace(-pi, pi, 45, endpoint=False)[1:]
    rot = special_ortho_group.rvs(3)

    for chi in chis:
        I, J, K, L = dihedral_points(chi)

        def _t(t):
            return torch.tensor(t).to(dtype=torch.double)

        gradcheck(
            VectorizedOp(geom.dihedral_angle_V_dV),
            (
                _t(I).requires_grad_(True),
                _t(J).requires_grad_(True),
                _t(K).requires_grad_(True),
                _t(L).requires_grad_(True),
            ),
        )

        # TODO flaky failures on some random rotations
        gradcheck(
            VectorizedOp(geom.dihedral_angle_V_dV),
            (
                _t(I @ rot).requires_grad_(True),
                _t(J @ rot).requires_grad_(True),
                _t(K @ rot).requires_grad_(True),
                _t(L @ rot).requires_grad_(True),
            ),
        )


@attr.s
class DihedralDat:
    coords = attr.ib()
    dihedral_atoms = attr.ib()
    dihedrals = attr.ib()


@pytest.fixture
def dihedral_test_data():
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
                    "0",
                    # fd: we clip acos to [-1,1] now
                    #     which means this does not return NaN
                    #     with current clipping (no fast-math) this returns 0.
                    #     ....but I'm not sure we should test this.
                    # no dispatching code should ever check this.
                ],
            )
        )
    )

    return DihedralDat(
        coords=coords, dihedral_atoms=dihedral_atoms, dihedrals=dihedrals
    )


def test_coord_dihedrals(geom, dihedral_test_data):
    coords, atoms, dihedrals = attr.astuple(dihedral_test_data)

    assert geom.dihedral_angle_V(
        coords[atoms.T[0]], coords[atoms.T[1]], coords[atoms.T[2]], coords[atoms.T[3]]
    ) == approx(dihedrals, nan_ok=True)

    assert geom.dihedral_angle_V_dV(
        coords[atoms.T[0]], coords[atoms.T[1]], coords[atoms.T[2]], coords[atoms.T[3]]
    )[0] == approx(dihedrals, nan_ok=True)


def test_coord_dihedral_angle_gradcheck(geom, dihedral_test_data):
    coords, atoms, dihedrals = attr.astuple(dihedral_test_data)
    # Remove last nan-valued entry for gradcheck.
    atoms = atoms[:-1]
    dihedrals = dihedrals[:-1]

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(geom.dihedral_angle_V_dV),
        (
            _t(coords[atoms.T[0]]).requires_grad_(True),
            _t(coords[atoms.T[1]]).requires_grad_(True),
            _t(coords[atoms.T[2]]).requires_grad_(True),
            _t(coords[atoms.T[3]]).requires_grad_(True),
        ),
    )
