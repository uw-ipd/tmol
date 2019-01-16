import pytest
from pytest import approx
import attr
from toolz import valmap, merge, keyfilter, keymap

import numpy
import torch
from tmol.tests.autograd import gradcheck, VectorizedOp
from tmol.utility.args import _signature, ignore_unused_kwargs


@pytest.fixture
def compiled(scope="session"):
    """Move compilation to test fixture to report compilation errors as test failure."""
    import tmol.score.ljlk.potentials.compiled

    return tmol.score.ljlk.potentials.compiled


def combine_params(params_i, params_j, global_params):
    return merge(
        keymap(lambda k: f"i_{k}", attr.asdict(params_i)),
        keymap(lambda k: f"j_{k}", attr.asdict(params_j)),
        attr.asdict(global_params),
    )


def test_lj_gradcheck(compiled, default_database):
    """Gradcheck lj_score_V_dV across range of value values."""
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
        sig = _signature(compiled.lj_score_V_dV)
        params = keyfilter(sig.parameters.__contains__, params)
        args = sig.bind(**valmap(_t, params)).arguments

        args["dist"] = args["dist"].requires_grad_(True)
        return tuple(args.values())

    op = VectorizedOp(compiled.lj_score_V_dV)
    kwargs = merge(
        dict(dist=torch.linspace(0, 8, 250), bonded_path_length=4),
        combine_params(i, j, g),
    )
    gradcheck(op, targs(kwargs), eps=5e-4)


def test_lj_spotcheck(compiled, default_database):
    """Check boundary conditionas and invarients in lj potential."""

    params = default_database.scoring.ljlk

    i = params.atom_type_parameters[0]
    j = params.atom_type_parameters[2]
    g = params.global_parameters

    sigma = i.lj_radius + j.lj_radius
    epsilon = numpy.sqrt(i.lj_wdepth * j.lj_wdepth)

    def eval_lj(dist, bonded_path_length=5):
        V, dV_dD = ignore_unused_kwargs(compiled.lj_score_V_dV)(
            dist, bonded_path_length, **combine_params(i, j, g)
        )

        return V

    def eval_lj_alone(dist, bonded_path_length=5):
        return ignore_unused_kwargs(compiled.lj_score_V)(
            dist, bonded_path_length, **combine_params(i, j, g)
        )

    def eval_d_lj_d_dist(dist, bonded_path_length=5):
        V, dV_dD = ignore_unused_kwargs(compiled.lj_score_V_dV)(
            dist, bonded_path_length, **combine_params(i, j, g)
        )

        return dV_dD

    # Linear region
    assert eval_lj(.6 * sigma - 1.0) == approx(
        eval_lj(.6 * sigma) - eval_d_lj_d_dist(.6 * sigma)
    )
    assert eval_d_lj_d_dist(numpy.linspace(0, .6 * sigma)) == approx(
        float(eval_d_lj_d_dist(.6 * sigma))
    )

    # Minimum value at sigma
    assert eval_lj(sigma) == approx(-epsilon)
    assert eval_lj(numpy.linspace(0, 8, 1000)).min() == approx(-epsilon)

    # Interpolate to 0
    assert eval_lj(4.5) == approx(compiled.vdw_V(4.5, sigma, epsilon))
    assert eval_lj(4.5) == approx(compiled.vdw_V_dV(4.5, sigma, epsilon)[0])
    assert eval_d_lj_d_dist(4.5) == approx(compiled.vdw_V_dV(4.5, sigma, epsilon)[1])

    assert eval_lj(6.0) == 0.0
    assert eval_d_lj_d_dist(6.0) == (0.0)

    # Bonded path length weights
    ds = numpy.linspace(0.0, 8.0, 100)
    numpy.testing.assert_allclose(eval_lj(ds, 4), eval_lj(ds, 5) * 0.2)
    numpy.testing.assert_allclose(eval_lj(ds, 2), 0.0)

    numpy.testing.assert_allclose(eval_lj(ds), eval_lj_alone(ds))
