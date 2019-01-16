import attr

import pytest
from pytest import approx

from toolz import keyfilter, keymap, valmap, merge

import torch
import numpy

from tmol.utility.args import ignore_unused_kwargs, _signature

from tmol.tests.autograd import gradcheck, VectorizedOp


@pytest.fixture
def compiled(scope="session"):
    """Move compilation to test fixture to report compilation errors as test failure."""
    import tmol.score.ljlk.potentials.compiled

    return tmol.score.ljlk.potentials.compiled


# TODO add lj_sigma spot check
parametrize_atom_pairs = pytest.mark.parametrize(
    "iname,jname", [("CNH2", "COO"), ("Ntrp", "OOC")]  # standard, donor/acceptor
)


def combine_params(params_i, params_j, global_params):
    return merge(
        keymap(lambda k: f"i_{k}", attr.asdict(params_i)),
        keymap(lambda k: f"j_{k}", attr.asdict(params_j)),
        attr.asdict(global_params),
    )


@pytest.mark.parametrize("bonded_path_length", [2, 4, 5])
@parametrize_atom_pairs
def test_lk_isotropic_gradcheck(
    compiled, default_database, iname, jname, bonded_path_length
):
    params = default_database.scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]
    g = params.global_parameters

    def _t(t):
        if isinstance(t, str):
            return

        t = torch.tensor(t)

        if t.is_floating_point():
            t = t.to(torch.double)

        return t

    def targs(params):
        sig = _signature(compiled.lk_isotropic_score_V_dV)
        params = keyfilter(sig.parameters.__contains__, params)
        args = sig.bind(**valmap(_t, params)).arguments

        args["dist"] = args["dist"].requires_grad_(True)
        return tuple(args.values())

    op = VectorizedOp(compiled.lk_isotropic_score_V_dV)
    kwargs = merge(
        dict(dist=torch.linspace(0, 8, 250), bonded_path_length=4),
        combine_params(i, j, g),
    )
    gradcheck(op, targs(kwargs))


@parametrize_atom_pairs
def test_lk_isotropic_spotcheck(compiled, default_database, iname, jname):
    params = default_database.scoring.ljlk

    i = {p.name: p for p in params.atom_type_parameters}[iname]
    j = {p.name: p for p in params.atom_type_parameters}[jname]
    g = params.global_parameters

    sigma = ignore_unused_kwargs(compiled.lj_sigma)(**combine_params(i, j, g))

    d_min = sigma * .89

    def eval_f_desolv(d):
        return compiled.f_desolv_V(
            d, i.lj_radius, i.lk_dgfree, i.lk_lambda, j.lk_volume
        ) + compiled.f_desolv_V(d, j.lj_radius, j.lk_dgfree, j.lk_lambda, i.lk_volume)

    def eval_lk_isotropic(d, bonded_path_length=5):
        return ignore_unused_kwargs(compiled.lk_isotropic_score_V)(
            d, bonded_path_length, **combine_params(i, j, g)
        )

    # Constant region
    assert eval_lk_isotropic(numpy.linspace(0, d_min - 0.25, 100)) == approx(
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
        (eval_f_desolv(d_min), eval_f_desolv(d_min + 0.25)), eval_lk_isotropic(d_min)
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
