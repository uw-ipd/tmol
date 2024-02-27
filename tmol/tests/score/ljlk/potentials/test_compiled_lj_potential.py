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
def test_lj_gradcheck(params, iname, jname, bonded_path_length):
    import tmol.tests.score.ljlk.potentials.compiled as compiled

    iidx, jidx = params.type_idx([iname, jname])
    i_params = params.type_params[iidx]
    j_params = params.type_params[jidx]

    global_params = params.global_params

    gradcheck(
        lambda dist: compiled.LJScore.apply(
            dist, bonded_path_length, i_params, j_params, global_params
        ).sum(),
        (torch.linspace(1, 8, 20, dtype=torch.double).requires_grad_(True),),
        eps=1e-4,
    )


@parametrize_atom_pairs
def test_lj_spotcheck(params, iname, jname):
    """Check boundary conditionas and invarients in lj potential."""

    import tmol.tests.score.ljlk.potentials.compiled as compiled

    iidx, jidx = params.type_idx([iname, jname])

    i = params.type_params[iidx]
    j = params.type_params[jidx]
    g = params.global_params

    sigma = compiled.lj_sigma(i, j, g)
    epsilon = numpy.sqrt(i.lj_wdepth * j.lj_wdepth).numpy()

    def eval_lj(dist, bonded_path_length=5):
        dist = (
            dist.clone().detach()
            if isinstance(dist, torch.Tensor)
            else torch.tensor(dist)
        )

        return (
            compiled.LJScore.apply(
                dist.requires_grad_(True), bonded_path_length, i, j, g
            )
            .detach()
            .numpy()
        )

    def eval_lj_alone(dist, bonded_path_length=5):
        dist = (
            dist.clone().detach()
            if isinstance(dist, torch.Tensor)
            else torch.tensor(dist)
        )

        return compiled.LJScore.apply(
            dist.requires_grad_(False), bonded_path_length, i, j, g
        ).numpy()

    def eval_d_lj_d_dist(dist, bonded_path_length=5):
        dist = (
            dist.clone().detach()
            if isinstance(dist, torch.Tensor)
            else torch.tensor(dist)
        )

        compiled.LJScore.apply(
            dist.requires_grad_(True), bonded_path_length, i, j, g
        ).sum().backward()

        return dist.grad.numpy()

    # Linear region
    assert eval_lj(0.6 * sigma - 1.0) == approx(
        eval_lj(0.6 * sigma) - eval_d_lj_d_dist(0.6 * sigma)
    )
    assert eval_d_lj_d_dist(numpy.linspace(0, 0.6 * sigma)) == approx(
        float(eval_d_lj_d_dist(0.6 * sigma))
    )

    # Minimum value at sigma
    assert eval_lj(sigma) == approx(-epsilon)
    assert eval_lj(numpy.linspace(0, 8, 1000)).min() == approx(
        -epsilon, abs=1e-5, rel=0
    )

    # Interpolate to 0
    assert eval_lj(4.5) == approx(compiled.vdw_V(4.5, sigma, epsilon))
    assert eval_lj(4.5) == approx(compiled.vdw_V_dV(4.5, sigma, epsilon)[0])
    assert eval_d_lj_d_dist(4.5) == approx(compiled.vdw_V_dV(4.5, sigma, epsilon)[1])

    assert eval_lj(6.0) == 0.0
    assert eval_d_lj_d_dist(6.0) == (0.0)

    # Bonded path length weights
    ds = torch.linspace(0.0, 8.0, 100)
    torch.testing.assert_close(eval_lj(ds, 4), eval_lj(ds, 5) * 0.2)
    assert (eval_lj(ds, 2) == 0.0).all()

    torch.testing.assert_close(eval_lj(ds), eval_lj_alone(ds))
