from pytest import approx

import torch

from numpy import array, cos, sin, linspace, pi
from scipy.stats import special_ortho_group

from tmol.tests.autograd import gradcheck, VectorizedOp


def _vecs(theta):
    return [[10.0, 0.0, 0.0], [2 * cos(theta), 2 * sin(theta), 0.0]]


# TODO Test cases for colinear and identical points.
def test_interior_angle_values():
    from tmol.score.common.geom import interior_angle_V_dV, interior_angle_V

    thetas = linspace(0, pi, 250, endpoint=False)
    rot = special_ortho_group.rvs(3)

    for theta in thetas:
        v1, v2 = _vecs(theta)

        assert interior_angle_V(v1, v2) == approx(theta)
        assert interior_angle_V_dV(v1, v2)[0] == approx(theta)

        assert interior_angle_V(v1 @ rot, v2 @ rot) == approx(theta)
        assert interior_angle_V_dV(v1 @ rot, v2 @ rot)[0] == approx(theta)


def test_interior_angle_gradcheck():
    from tmol.score.common.geom import interior_angle_V_dV

    thetas = linspace(1e-5, pi, 100, endpoint=False)
    v1, v2 = array(list(map(_vecs, thetas))).swapaxes(0, 1)

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(interior_angle_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
    )


def test_cos_interior_angle_values():
    from tmol.score.common.geom import cos_interior_angle_V_dV, cos_interior_angle_V

    thetas = linspace(0, pi, 250, endpoint=False)
    rot = special_ortho_group.rvs(3)

    for theta in thetas:
        v1, v2 = _vecs(theta)

        assert cos_interior_angle_V(v1, v2) == approx(cos(theta))
        assert cos_interior_angle_V_dV(v1, v2)[0] == approx(cos(theta))

        assert cos_interior_angle_V(v1 @ rot, v2 @ rot) == approx(cos(theta))
        assert cos_interior_angle_V_dV(v1 @ rot, v2 @ rot)[0] == approx(cos(theta))


def test_cos_interior_angle_gradcheck():
    from tmol.score.common.geom import cos_interior_angle_V_dV

    thetas = linspace(1e-5, pi, 100, endpoint=False)
    v1, v2 = array(list(map(_vecs, thetas))).swapaxes(0, 1)

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(cos_interior_angle_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
    )
