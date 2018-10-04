import torch

from tmol.database import ParameterDatabase
from tmol.score.ljlk.params import LJLKParamResolver


def test_lj_welldepth():
    import tmol.score.ljlk.cpp_potential as cpp_potential

    params = LJLKParamResolver.from_database(
        ParameterDatabase.get_default().scoring.ljlk, torch.device("cpu")
    )

    at = params.type_params[1]
    bt = params.type_params[8]

    pparams = params.pair_params[1, 8]

    dmin = at.lj_radius + bt.lj_radius
    depth = -(at.lj_wdepth + bt.lj_wdepth) / 2

    pval = cpp_potential.cpu.lj_potential(
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
