import pytest
from pytest import approx

import torch
import numpy

from tmol.tests.autograd import gradcheck

from tmol.score.ljlk.params import LJLKParamResolver


@pytest.fixture
def params(default_database):
    return LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch.device("cpu")
    )


parametrize_atom_pairs = pytest.mark.parametrize(
    "iname,jname", [("CNH2", "COO"), ("Ntrp", "OOC")]  # standard, donor/acceptor
)


@pytest.mark.parametrize("bonded_path_length", [2, 4, 5])
@parametrize_atom_pairs
def test_lk_isotropic_gradcheck(params, iname, jname, bonded_path_length):
    import tmol.tests.score.ljlk.potentials.compiled as compiled

    iidx, jidx = params.type_idx([iname, jname])
    i_params = params.type_params[iidx]
    j_params = params.type_params[jidx]

    global_params = params.global_params

    gradcheck(
        lambda dist: compiled.LKScore.apply(
            dist, bonded_path_length, i_params, j_params, global_params
        ).sum(),
        (torch.linspace(1, 8, 20, dtype=torch.double).requires_grad_(True),),
        eps=3e-4,
        atol=1e-5,
        rtol=1e-3,
    )


@parametrize_atom_pairs
def test_lk_spotcheck(params, iname, jname):
    """Check boundary conditionas and invarients in lj potential."""

    import tmol.tests.score.ljlk.potentials.compiled as compiled

    iidx, jidx = params.type_idx([iname, jname])

    i = params.type_params[iidx]
    j = params.type_params[jidx]
    g = params.global_params

    sigma = compiled.lj_sigma(i, j, g)

    d_min = sigma * 0.89
    cpoly_close_dmin = numpy.sqrt(d_min * d_min - 1.45)
    cpoly_close_dmax = numpy.sqrt(d_min * d_min + 1.05)

    def eval_f_desolv(d):
        return compiled.f_desolv_V(
            d, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume
        ) + compiled.f_desolv_V(d, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume)

    def eval_lk_isotropic(d, bonded_path_length=5):
        return compiled.LKScore.apply(
            torch.tensor(d), bonded_path_length, i, j, g
        ).numpy()

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
    assert eval_lk_isotropic(ds, 4) == approx(eval_lk_isotropic(ds, 5) * 0.2)
    assert eval_lk_isotropic(ds, 2) == approx(0.0, rel=0, abs=0)
