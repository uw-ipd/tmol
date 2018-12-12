from pytest import approx

import numpy
import scipy.optimize

import tmol.score.vdw.numba.lj
from tmol.score.vdw.numba.lj import lj, d_lj_d_dist, f_vdw, f_vdw_d_dist

import tmol.database


def test_lj_gradcheck():
    params = tmol.database.ParameterDatabase.get_default().scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    ds = numpy.linspace(0, 10, 1000)

    sigma = i.lj_radius + j.lj_radius

    grad_errors = numpy.array(
        [
            scipy.optimize.check_grad(
                lj,
                d_lj_d_dist,
                numpy.array([d]),
                i.lj_radius,
                i.lj_wdepth,
                i.is_donor,
                i.is_hydroxyl,
                i.is_polarh,
                i.is_acceptor,
                j.lj_radius,
                j.lj_wdepth,
                j.is_donor,
                j.is_hydroxyl,
                j.is_polarh,
                j.is_acceptor,
                params.global_parameters.lj_hbond_dis,
                params.global_parameters.lj_hbond_OH_donor_dis,
                params.global_parameters.lj_hbond_hdis,
            )
            for d in ds
        ]
    )

    # Reduce grad check precision in repulsive regime due to high magnitude derivs
    numpy.testing.assert_allclose(grad_errors[ds < sigma], 0, atol=1e-5)
    numpy.testing.assert_allclose(grad_errors[ds > sigma], 0, atol=1e-7)


def test_lj_spotcheck():
    params = tmol.database.ParameterDatabase.get_default().scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    sigma = i.lj_radius + j.lj_radius
    epsilon = numpy.sqrt(i.lj_wdepth * j.lj_wdepth)

    def eval_lj(d):
        return lj(
            d,
            i.lj_radius,
            i.lj_wdepth,
            i.is_donor,
            i.is_hydroxyl,
            i.is_polarh,
            i.is_acceptor,
            j.lj_radius,
            j.lj_wdepth,
            j.is_donor,
            j.is_hydroxyl,
            j.is_polarh,
            j.is_acceptor,
            params.global_parameters.lj_hbond_dis,
            params.global_parameters.lj_hbond_OH_donor_dis,
            params.global_parameters.lj_hbond_hdis,
        )

    def eval_d_lj_d_dist(d):
        return d_lj_d_dist(
            d,
            i.lj_radius,
            i.lj_wdepth,
            i.is_donor,
            i.is_hydroxyl,
            i.is_polarh,
            i.is_acceptor,
            j.lj_radius,
            j.lj_wdepth,
            j.is_donor,
            j.is_hydroxyl,
            j.is_polarh,
            j.is_acceptor,
            params.global_parameters.lj_hbond_dis,
            params.global_parameters.lj_hbond_OH_donor_dis,
            params.global_parameters.lj_hbond_hdis,
        )

    # Linear region
    assert eval_lj(.6 * sigma - 1.0) == approx(
        eval_lj(.6 * sigma) - eval_d_lj_d_dist(.6 * sigma)
    )
    assert eval_d_lj_d_dist(numpy.linspace(0, .6 * sigma)) == approx(
        eval_d_lj_d_dist(.6 * sigma)
    )

    # Minimum value at sigma
    assert eval_lj(sigma) == approx(-epsilon)
    assert numpy.all(eval_lj(numpy.linspace(0, 8, 1000)) > -epsilon)

    # Interpolate to 0
    assert eval_lj(4.5) == approx(f_vdw(4.5, sigma, epsilon))
    assert eval_d_lj_d_dist(4.5) == approx(f_vdw_d_dist(4.5, sigma, epsilon))
    assert eval_lj(6.0) == 0.0
    assert eval_d_lj_d_dist(6.0) == (0.0)
