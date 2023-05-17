import pytest
from pytest import approx

import torch
import numpy
import scipy.optimize


from tmol.score.ljlk.params import LJLKParamResolver
from tmol.tests.numba import requires_numba_jit


@pytest.fixture
def params(default_database):
    return LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch.device("cpu")
    )


@requires_numba_jit
@pytest.mark.parametrize("bonded_path_length", [2, 4, 5])
def test_lj_gradcheck(params, bonded_path_length):
    from tmol.score.ljlk.numba.vectorized import lj, d_lj_d_dist

    i = params.type_params[0]
    j = params.type_params[2]
    g = params.global_params

    ds = numpy.linspace(0, 10, 1000)

    bonded_path_length = 4

    sigma = (i.lj_radius + j.lj_radius).numpy()

    grad_errors = numpy.array(
        [
            scipy.optimize.check_grad(
                lj,
                d_lj_d_dist,
                numpy.array([d]),
                bonded_path_length,
                i.lj_radius.numpy(),
                i.lj_wdepth.numpy(),
                i.is_donor.numpy(),
                i.is_hydroxyl.numpy(),
                i.is_polarh.numpy(),
                i.is_acceptor.numpy(),
                j.lj_radius.numpy(),
                j.lj_wdepth.numpy(),
                j.is_donor.numpy(),
                j.is_hydroxyl.numpy(),
                j.is_polarh.numpy(),
                j.is_acceptor.numpy(),
                g.lj_hbond_dis.numpy(),
                g.lj_hbond_OH_donor_dis.numpy(),
                g.lj_hbond_hdis.numpy(),
            )
            for d in ds
        ]
    )

    # Reduce grad check precision in repulsive regime due to high magnitude derivs
    numpy.testing.assert_allclose(grad_errors[ds < sigma], 0, atol=1e-5)
    numpy.testing.assert_allclose(grad_errors[ds > sigma], 0, atol=1e-6)


@requires_numba_jit
def test_lj_spotcheck(params):
    from tmol.score.ljlk.numba.lj import f_vdw, f_vdw_d_dist
    from tmol.score.ljlk.numba.vectorized import lj, d_lj_d_dist

    i = params.type_params[0]
    j = params.type_params[2]
    g = params.global_params

    sigma = (i.lj_radius + j.lj_radius).numpy()
    epsilon = numpy.sqrt(i.lj_wdepth * j.lj_wdepth).numpy()

    def eval_lj(d, bonded_path_length=5):
        return lj(
            d,
            bonded_path_length,
            i.lj_radius.numpy(),
            i.lj_wdepth.numpy(),
            i.is_donor.numpy(),
            i.is_hydroxyl.numpy(),
            i.is_polarh.numpy(),
            i.is_acceptor.numpy(),
            j.lj_radius.numpy(),
            j.lj_wdepth.numpy(),
            j.is_donor.numpy(),
            j.is_hydroxyl.numpy(),
            j.is_polarh.numpy(),
            j.is_acceptor.numpy(),
            g.lj_hbond_dis.numpy(),
            g.lj_hbond_OH_donor_dis.numpy(),
            g.lj_hbond_hdis.numpy(),
        )

    def eval_d_lj_d_dist(d, bonded_path_length=5):
        return d_lj_d_dist(
            d,
            bonded_path_length,
            i.lj_radius.numpy(),
            i.lj_wdepth.numpy(),
            i.is_donor.numpy(),
            i.is_hydroxyl.numpy(),
            i.is_polarh.numpy(),
            i.is_acceptor.numpy(),
            j.lj_radius.numpy(),
            j.lj_wdepth.numpy(),
            j.is_donor.numpy(),
            j.is_hydroxyl.numpy(),
            j.is_polarh.numpy(),
            j.is_acceptor.numpy(),
            g.lj_hbond_dis.numpy(),
            g.lj_hbond_OH_donor_dis.numpy(),
            g.lj_hbond_hdis.numpy(),
        )

    # Linear region
    assert eval_lj(0.6 * sigma - 1.0) == approx(
        eval_lj(0.6 * sigma) - eval_d_lj_d_dist(0.6 * sigma)
    )
    assert eval_d_lj_d_dist(numpy.linspace(0, 0.6 * sigma, 50)) == approx(
        eval_d_lj_d_dist(0.6 * sigma).repeat(50)
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
