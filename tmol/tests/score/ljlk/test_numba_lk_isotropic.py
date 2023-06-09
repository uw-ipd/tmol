import attr
import toolz

import pytest
from pytest import approx

import torch

import numpy
import scipy.optimize

from tmol.utility.args import ignore_unused_kwargs
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.tests.numba import requires_numba_jit


# TODO add lj_sigma spot check
parametrize_atom_pairs = pytest.mark.parametrize(
    "iname,jname", [("CNH2", "COO"), ("Ntrp", "OOC")]  # standard, donor/acceptor
)


def combine_params(params_i, params_j, global_params):
    def _t(t):
        # Coerce 0-d to scalars for numba
        if t.dim() == 0:
            return t.numpy()[()]
        if t.dim() == 1 and t.shape == (1,):
            return t.numpy()[0]
        else:
            return t.numpy()

    return toolz.merge(
        {f"{k}_i": _t(t) for k, t in attr.asdict(params_i).items()},
        {f"{k}_j": _t(t) for k, t in attr.asdict(params_j).items()},
        {k: _t(t) for k, t in attr.asdict(global_params).items()},
    )


@pytest.fixture
def params(default_database):
    return LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch.device("cpu")
    )


@requires_numba_jit
@pytest.mark.parametrize("bonded_path_length", [2, 4, 5])
@parametrize_atom_pairs
def test_lk_isotropic_gradcheck(params, iname, jname, bonded_path_length):
    from tmol.score.ljlk.numba.vectorized import lk_isotropic, d_lk_isotropic_d_dist

    iidx, jidx = params.type_idx([iname, jname])
    i = params.type_params[iidx]
    j = params.type_params[jidx]
    g = params.global_params

    ds = numpy.linspace(0, 10, 1000)

    # Bind parameter values via function for checkgrad.
    def eval_f(d):
        return ignore_unused_kwargs(lk_isotropic)(
            d, bonded_path_length, **combine_params(i, j, g)
        )

    def eval_f_d_d(d):
        return ignore_unused_kwargs(d_lk_isotropic_d_dist)(
            d, bonded_path_length, **combine_params(i, j, g)
        )

    grad_errors = numpy.array(
        [scipy.optimize.check_grad(eval_f, eval_f_d_d, numpy.array([d])) for d in ds]
    )

    numpy.testing.assert_allclose(grad_errors, 0, atol=1e-7)


@requires_numba_jit
@parametrize_atom_pairs
def test_lk_isotropic_spotcheck(params, iname, jname):
    from tmol.score.ljlk.numba.lk_isotropic import f_desolv
    from tmol.score.ljlk.numba.common import lj_sigma
    from tmol.score.ljlk.numba.vectorized import lk_isotropic

    iidx, jidx = params.type_idx([iname, jname])
    i = params.type_params[iidx]
    j = params.type_params[jidx]
    g = params.global_params

    sigma = ignore_unused_kwargs(lj_sigma)(**combine_params(i, j, g))

    d_min = sigma * 0.89
    cpoly_close_dmin = numpy.sqrt(d_min * d_min - 1.45)
    cpoly_close_dmax = numpy.sqrt(d_min * d_min + 1.05)

    def eval_f_desolv(d):
        return f_desolv(
            d,
            i.lj_radius.numpy(),
            i.lk_dgfree.numpy(),
            i.lk_lambda.numpy(),
            j.lk_volume.numpy(),
        ) + f_desolv(
            d,
            j.lj_radius.numpy(),
            j.lk_dgfree.numpy(),
            j.lk_lambda.numpy(),
            i.lk_volume.numpy(),
        )

    def eval_lk_isotropic(d, bonded_path_length=5):
        return ignore_unused_kwargs(lk_isotropic)(
            d, bonded_path_length, **combine_params(i, j, g)
        )

    # Constant region
    assert eval_lk_isotropic(numpy.linspace(0, cpoly_close_dmin, 100)) == approx(
        eval_f_desolv(d_min)
    )

    def is_between(a_b, x):
        a, b = a_b
        if a < b:
            return a <= x and x <= b
        else:
            return b <= x and x <= a

    # Interpolate to f(d_min)
    assert is_between(
        (eval_f_desolv(d_min), eval_f_desolv(cpoly_close_dmax)),
        eval_lk_isotropic(d_min),
    )

    # Interpolate to 0
    assert eval_lk_isotropic(4.5) == approx(eval_f_desolv(4.5))

    # Interpolate to 0
    assert eval_lk_isotropic(6.0) == approx(0.0)
    assert eval_lk_isotropic(8.0) == approx(0.0)

    # Bonded path length weights
    ds = numpy.linspace(0.0, 8.0, 100)
    numpy.testing.assert_allclose(
        eval_lk_isotropic(ds, 4), eval_lk_isotropic(ds, 5) * 0.2
    )
    numpy.testing.assert_allclose(eval_lk_isotropic(ds, 2), 0.0)
