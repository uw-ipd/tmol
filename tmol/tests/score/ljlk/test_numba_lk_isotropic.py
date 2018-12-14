from pytest import approx

import numpy
import scipy.optimize

import tmol.score.ljlk.numba.lk_isotropic
from tmol.score.ljlk.numba.lk_isotropic import (
    # lk_pair_total,
    f_desolv,
    lk_isotropic_mutual,
    d_lk_isotropic_mutual_d_dist,
)

import tmol.database


def test_lk_isotropic_gradcheck():
    params = tmol.database.ParameterDatabase.get_default().scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    ds = numpy.linspace(0, 10, 1000)

    grad_errors = numpy.array(
        [
            scipy.optimize.check_grad(
                lk_isotropic_mutual,
                d_lk_isotropic_mutual_d_dist,
                numpy.array([d]),
                i.lj_radius,
                i.lk_dgfree,
                i.lk_lambda,
                i.lj_radius,
                j.lj_radius,
                j.lk_dgfree,
                j.lk_lambda,
                j.lj_radius,
            )
            for d in ds
        ]
    )

    numpy.testing.assert_allclose(grad_errors, 0, atol=1e-7)


def test_lk_isotropic_spotcheck():
    params = tmol.database.ParameterDatabase.get_default().scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    sigma = i.lj_radius + j.lj_radius

    def eval_f_desolv(d):
        return f_desolv(
            d, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume
        ) + f_desolv(d, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume)

    def eval_lk_mutual(d):
        return lk_isotropic_mutual(
            d,
            i.lj_radius,
            i.lk_dgfree,
            i.lk_lambda,
            i.lk_volume,
            j.lj_radius,
            j.lk_dgfree,
            j.lk_lambda,
            j.lk_volume,
        )

    # Constant region
    assert eval_lk_mutual(numpy.linspace(0, sigma - 0.2, 100)) == approx(
        eval_f_desolv(sigma)
    )

    # Interpolate to f(sigma)
    assert (eval_lk_mutual(sigma) < eval_f_desolv(sigma)) and (
        eval_lk_mutual(sigma) > eval_f_desolv(sigma + 0.3)
    )

    # Interpolate to 0
    assert eval_lk_mutual(4.5) == approx(eval_f_desolv(4.5))

    # Interpolate to 0
    assert eval_lk_mutual(6.0) == approx(0.0)
    assert eval_lk_mutual(8.0) == approx(0.0)
