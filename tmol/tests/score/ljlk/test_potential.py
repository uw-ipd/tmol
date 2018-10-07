import functools

import numpy
import scipy.optimize
import pandas

import torch

from tmol.utility.reactive import reactive_attrs
from tmol.tests.benchmark import subfixture

from tmol.database import ParameterDatabase
from tmol.score.ljlk.params import LJLKParamResolver

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.bonded_atom import BondedAtomScoreGraph

import tmol.score.ljlk.torch_potential as torch_potential


def test_lj_welldepth_smoketest():
    """Check cpp potential against expected value.

    Value reported by the cpp potential implementation should match the
    expected well depth from atom type paramters.
    """
    import tmol.score.ljlk.cpp_potential as cpp_potential

    params = LJLKParamResolver.from_database(
        ParameterDatabase.get_default().scoring.ljlk, torch.device("cpu")
    )

    at = params.type_params[1]
    bt = params.type_params[8]

    pparams = params.pair_params[1, 8]

    dmin = at.lj_radius + bt.lj_radius
    depth = -(at.lj_wdepth + bt.lj_wdepth) / 2

    pval = cpp_potential.cpu.lj(
        dmin,
        10,
        pparams.lj_sigma,
        pparams.lj_switch_slope,
        pparams.lj_switch_intercept,
        pparams.lj_coeff_sigma12,
        pparams.lj_coeff_sigma6,
        pparams.lj_spline_y0,
        pparams.lj_spline_dy0,
        params.global_params.lj_switch_dis2sigma,
        params.global_params.spline_start,
        params.global_params.max_dis,
    )

    torch.testing.assert_allclose(depth, pval, rtol=5e-3, atol=0)


def lj_gradcheck():
    """Generate gradcheck summary frame for lj potential.

    See dependent analysis in dev/score/ljlj/lj_backward.
    """
    import tmol.score.ljlk.cpp_potential as cpp_potential

    lj = numpy.vectorize(cpp_potential.cpu.lj)
    d_lj_d_dist = numpy.vectorize(cpp_potential.cpu.d_lj_d_dist)

    params = LJLKParamResolver.from_database(
        ParameterDatabase.get_default().scoring.ljlk, torch.device("cpu")
    )

    pparams = params.pair_params[1, 8]

    input_params = (
        numpy.array([10], dtype="u1"),
        pparams.lj_sigma,
        pparams.lj_switch_slope,
        pparams.lj_switch_intercept,
        pparams.lj_coeff_sigma12,
        pparams.lj_coeff_sigma6,
        pparams.lj_spline_y0,
        pparams.lj_spline_dy0,
        params.global_params.lj_switch_dis2sigma,
        params.global_params.spline_start,
        params.global_params.max_dis,
    )

    dist = numpy.linspace(0, 6.1, 100)

    numeric = numpy.concatenate(
        [scipy.optimize.approx_fprime([p], lj, 1e-4, *input_params) for p in dist]
    )
    analytic = numpy.concatenate([d_lj_d_dist(p, *input_params) for p in dist])

    gradcheck = pandas.DataFrame.from_dict(
        dict(dist=dist, numeric=numeric, analytic=analytic)
    )
    gradcheck = gradcheck.eval("absolute = numeric - analytic")
    gradcheck = gradcheck.eval("relative = (numeric - analytic) / analytic")

    return gradcheck


def test_lj_gradcheck():
    """Gradcheck lj potential wrt distance."""

    gradcheck = lj_gradcheck()

    torch.testing.assert_allclose(
        gradcheck.numeric, gradcheck.analytic, rtol=7.5e-3, atol=1e-4
    )


def test_cpp_torch_potential_comparison(benchmark, ubq_system, torch_device):
    import tmol.score.ljlk.cpp_potential as cpp_potential

    @reactive_attrs
    class DataGraph(CartesianAtomicCoordinateProvider, BondedAtomScoreGraph):
        pass

    ubq_g = DataGraph.build_for(ubq_system, device=torch_device)

    params = LJLKParamResolver.from_database(
        ParameterDatabase.get_default().scoring.ljlk, ubq_g.device
    )

    coords = ubq_g.coords[0].detach()
    type_strs = ubq_g.atom_types[0]
    bonded_path_length = torch.tensor(ubq_g.bonded_path_length[0])

    bonded_path_length[bonded_path_length > 6] = 255
    bonded_path_length = bonded_path_length.to(device=torch_device, dtype=torch.uint8)

    types = params.type_idx(type_strs)
    types[type_strs == None] = -1  # noqa
    types = torch.tensor(types).to(device=torch_device)

    # prev score form
    a = coords[:, None]
    a_t = types[:, None]
    b = coords[None, :]
    b_t = types[None, :]

    pparams = params.pair_params[a_t, b_t]

    assert coords.device == torch_device

    def bsync(f):
        if torch_device.type == "cuda":

            @functools.wraps(f)
            def syncf():
                try:
                    return f()
                finally:
                    torch.cuda.synchronize()

            return syncf
        else:
            return f

    @subfixture(benchmark)
    @bsync
    def torch_impl():
        delta = (a[..., 0] - b[..., 0], a[..., 1] - b[..., 1], a[..., 2] - b[..., 2])
        dists = torch.sqrt(
            delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
        )

        pscore = torch_potential.lj_score(
            dists,
            bonded_path_length,
            lj_sigma=pparams.lj_sigma,
            lj_switch_slope=pparams.lj_switch_slope,
            lj_switch_intercept=pparams.lj_switch_intercept,
            lj_coeff_sigma12=pparams.lj_coeff_sigma12,
            lj_coeff_sigma6=pparams.lj_coeff_sigma6,
            lj_spline_y0=pparams.lj_spline_y0,
            lj_spline_dy0=pparams.lj_spline_dy0,
            # Global params
            lj_switch_dis2sigma=params.global_params.lj_switch_dis2sigma,
            spline_start=params.global_params.spline_start,
            max_dis=params.global_params.max_dis,
        )

        return torch.triu(pscore, diagonal=1)

    torch_impl[torch.isnan(torch_impl)] = 0.0

    @subfixture(benchmark)
    @bsync
    def cpp_impl():
        return cpp_potential.lj_intra(
            coords,
            types,
            bonded_path_length,
            lj_sigma=params.pair_params.lj_sigma,
            lj_switch_slope=params.pair_params.lj_switch_slope,
            lj_switch_intercept=params.pair_params.lj_switch_intercept,
            lj_coeff_sigma12=params.pair_params.lj_coeff_sigma12,
            lj_coeff_sigma6=params.pair_params.lj_coeff_sigma6,
            lj_spline_y0=params.pair_params.lj_spline_y0,
            lj_spline_dy0=params.pair_params.lj_spline_dy0,
            # Global params
            lj_switch_dis2sigma=params.global_params.lj_switch_dis2sigma,
            spline_start=params.global_params.spline_start,
            max_dis=params.global_params.max_dis,
        )

    torch.testing.assert_allclose(torch_impl, cpp_impl)
