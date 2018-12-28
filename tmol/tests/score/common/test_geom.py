from pytest import approx

import torch

from toolz import merge, curry
from numpy import array, dot, arctan2, cos, sin, cross, linspace, pi
from numpy.linalg import norm
from scipy.stats import special_ortho_group
import pandas

from tmol.tests.autograd import gradcheck, VectorizedOp


def _vecs(theta):
    return [[10.0, 0.0, 0.0], [2 * cos(theta), 2 * sin(theta), 0.0]]


# TODO Test cases for colinear and identical points.
def test_interior_angle_values():
    from tmol.score.common.geom import interior_angle_V_dV, interior_angle_V

    def angles(vs):
        v1, v2 = vs
        c = cross(v1, v2)
        return dict(
            arctan2=2 * arctan2(dot(c, c / norm(c)), norm(v1) * norm(v2) + dot(v1, v2)),
            compiled_V_dV=float(interior_angle_V_dV(v1, v2)[0]),
            compiled_V=interior_angle_V(v1, v2),
            v1=v1,
            v2=v2,
        )

    @curry
    def tentry(theta, rot):
        return merge(angles(_vecs(theta) @ rot), dict(theta=theta))

    rframe = pandas.DataFrame.from_records(
        map(
            tentry(rot=special_ortho_group.rvs(3)), linspace(0, pi, 250, endpoint=False)
        )
    )

    assert array(rframe.compiled_V_dV) == approx(array(rframe.theta))
    assert array(rframe.compiled_V) == approx(array(rframe.theta))


def test_interior_angle_gradcheck():
    from tmol.score.common.geom import interior_angle_V_dV

    v1, v2 = zip(*map(_vecs, linspace(1e-5, pi, 100, endpoint=False)))

    def _t(t):
        return torch.tensor(t).to(dtype=torch.double)

    gradcheck(
        VectorizedOp(interior_angle_V_dV),
        (_t(array(v1)).requires_grad_(True), _t(array(v2)).requires_grad_(True)),
    )
