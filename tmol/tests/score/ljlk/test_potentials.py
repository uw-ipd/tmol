import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.tests.benchmark import subfixture

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.bonded_atom import BondedAtomScoreGraph

import tmol.database
from tmol.score.ljlk.params import LJLKParamResolver

import tmol.score.ljlk.potentials as potentials
import tmol.score.ljlk.numba_potential as numba_potential


@reactive_attrs
class DataGraph(CartesianAtomicCoordinateProvider, BondedAtomScoreGraph):
    pass


def test_potential_comparisons(benchmark, ubq_system, torch_device):
    ubq_g = DataGraph.build_for(ubq_system, device=torch_device)

    params = LJLKParamResolver.from_database(
        tmol.database.ParameterDatabase.get_default().scoring.ljlk, ubq_g.device
    )

    coords = ubq_g.coords[0].detach()
    type_strs = ubq_g.atom_types[0]
    bonded_path_length = torch.tensor(ubq_g.bonded_path_length[0])

    bonded_path_length[bonded_path_length > 6] = 255
    bonded_path_length = bonded_path_length.to(device=torch_device, dtype=torch.uint8)

    types = params.type_idx(type_strs)
    types[type_strs == None] = -1  # noqa

    # prev score form
    a = coords[:, None]
    a_t = types[:, None]
    b = coords[None, :]
    b_t = types[None, :]

    pparams = params.pair_params[a_t, b_t]

    assert coords.device == torch_device

    @subfixture(benchmark)
    def torch_impl():
        delta = (a[..., 0] - b[..., 0], a[..., 1] - b[..., 1], a[..., 2] - b[..., 2])
        dists = torch.sqrt(
            delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
        )

        pscore = potentials.lj_score(
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

    # new op
    @subfixture(benchmark)
    def numba_impl_serial():
        return numba_potential.lj_intra_kernel(
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
            parallel=False,
        )

    @subfixture(benchmark)
    def numba_impl():
        return numba_potential.lj_intra_kernel(
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
            parallel=True,
        )

    torch.testing.assert_allclose(torch_impl, numba_impl)
    torch.testing.assert_allclose(torch_impl, numba_impl_serial)


# def test_potential_trace():
#     ljlk_resolver = tmol.score.ljlk.params.LJLKParamResolver.from_database(
#         tmol.database.ParameterDatabase.get_default().scoring.ljlk, torch.device("cpu")
#     )
#     ljlk_params = dict(
#         lj_sigma=ljlk_resolver.pair_params.lj_sigma,
#         lj_switch_slope=ljlk_resolver.pair_params.lj_switch_slope,
#         lj_switch_intercept=ljlk_resolver.pair_params.lj_switch_intercept,
#         lj_coeff_sigma12=ljlk_resolver.pair_params.lj_coeff_sigma12,
#         lj_coeff_sigma6=ljlk_resolver.pair_params.lj_coeff_sigma6,
#         lj_spline_y0=ljlk_resolver.pair_params.lj_spline_y0,
#         lj_spline_dy0=ljlk_resolver.pair_params.lj_spline_dy0,
#         lj_switch_dis2sigma=ljlk_resolver.global_params.lj_switch_dis2sigma,
#         spline_start=ljlk_resolver.global_params.spline_start,
#         max_dis=ljlk_resolver.global_params.max_dis,
#     )

#     x = torch.zeros((1, 3))
#     y = torch.zeros((1000, 3))
#     y[:, 0] = torch.linspace(.1, ljlk_resolver.global_params.max_dis + 1, 1000)

#     xt = torch.full((x.shape[0],), 1, dtype=torch.int64)
#     yt = torch.full((y.shape[0],), 1, dtype=torch.int64)
#     news = tmol.score.ljlk.numba_potential.lj_kernel(
#         x, xt, y, yt, torch.full((1, 1000), 10, dtype=torch.uint8), **ljlk_params
#     )

#     olds = tmol.score.ljlk.potentials.lj_score(
#         y[:, 0],
#         torch.full((1, 1000), 10, dtype=torch.uint8),
#         lj_sigma=ljlk_resolver.pair_params.lj_sigma[xt, yt],
#         lj_switch_slope=ljlk_resolver.pair_params.lj_switch_slope[xt, yt],
#         lj_switch_intercept=ljlk_resolver.pair_params.lj_switch_intercept[xt, yt],
#         lj_coeff_sigma12=ljlk_resolver.pair_params.lj_coeff_sigma12[xt, yt],
#         lj_coeff_sigma6=ljlk_resolver.pair_params.lj_coeff_sigma6[xt, yt],
#         lj_spline_y0=ljlk_resolver.pair_params.lj_spline_y0[xt, yt],
#         lj_spline_dy0=ljlk_resolver.pair_params.lj_spline_dy0[xt, yt],
#         lj_switch_dis2sigma=ljlk_resolver.global_params.lj_switch_dis2sigma,
#         spline_start=ljlk_resolver.global_params.spline_start,
#         max_dis=ljlk_resolver.global_params.max_dis,
#     )

#     numpy.testing.assert_allclose(news, olds)
