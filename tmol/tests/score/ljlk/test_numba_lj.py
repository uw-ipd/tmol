import pytest
from pytest import approx

import numpy
import scipy.optimize

from tmol.score.ljlk.numba.lj import f_vdw, f_vdw_d_dist
from tmol.score.ljlk.numba.vectorized import lj, d_lj_d_dist


@pytest.mark.parametrize("bonded_path_length", [2, 4, 5])
def test_lj_gradcheck(default_database, bonded_path_length):
    params = default_database.scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    ds = numpy.linspace(0, 10, 1000)

    bonded_path_length = 4

    sigma = i.lj_radius + j.lj_radius

    grad_errors = numpy.array(
        [
            scipy.optimize.check_grad(
                lj,
                d_lj_d_dist,
                numpy.array([d]),
                bonded_path_length,
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


def test_lj_spotcheck(default_database):
    params = default_database.scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]

    sigma = i.lj_radius + j.lj_radius
    epsilon = numpy.sqrt(i.lj_wdepth * j.lj_wdepth)

    def eval_lj(d, bonded_path_length=5):
        return lj(
            d,
            bonded_path_length,
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

    def eval_d_lj_d_dist(d, bonded_path_length=5):
        return d_lj_d_dist(
            d,
            bonded_path_length,
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

    # Bonded path length weights
    ds = numpy.linspace(0.0, 8.0, 100)
    numpy.testing.assert_allclose(eval_lj(ds, 4), eval_lj(ds, 5) * 0.2)
    numpy.testing.assert_allclose(eval_lj(ds, 2), 0.0)
