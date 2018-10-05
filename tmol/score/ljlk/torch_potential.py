import torch

from tmol.types.functional import validate_args

from tmol.types.torch import Tensor

from .params import LJLKGlobalParams, LJLKTypeParams, LJLKTypePairParams

Params = Tensor(torch.float)[...]


@validate_args
def render_pair_parameters(
    global_params: LJLKGlobalParams, params_a: LJLKTypeParams, params_b: LJLKTypeParams
) -> LJLKTypePairParams:

    # Simplified pair broadcast, no implicit dimensions are added.  Allocate a
    # result buffer the size of the direct broadcast of the two input parameter sets.
    assert len(params_a.shape) == len(params_b.shape)
    result_shape = [max(d) for d in zip(params_a.shape, params_b.shape)]
    params_p = LJLKTypePairParams.empty(
        result_shape, device=global_params.max_dis.device
    )

    # lj
    # these are only dependent on atom1/atom2 ... can this be more efficient?
    params_p.lj_rad1[:] = params_a.lj_radius
    params_p.lj_rad2[:] = params_b.lj_radius

    sigma = params_a.lj_radius + params_b.lj_radius

    # exception 1: acc :: non-OH donor radii
    don_acc_pair_mask = (
        params_a.is_donor & ~params_a.is_hydroxyl & params_b.is_acceptor
    ) | (params_b.is_donor & ~params_b.is_hydroxyl & params_a.is_acceptor)
    sigma[don_acc_pair_mask] = global_params.lj_hbond_dis  # lj_hbond_dis

    # exception 2: acc :: OH donor radii
    don_acc_pair_mask = (
        params_a.is_donor & params_a.is_hydroxyl & params_b.is_acceptor
    ) | (params_b.is_donor & params_b.is_hydroxyl & params_a.is_acceptor)
    sigma[don_acc_pair_mask] = global_params.lj_hbond_OH_donor_dis

    # exception 3: acc :: donor H radii
    don_acc_pair_mask = (params_a.is_polarh & params_b.is_acceptor) | (
        params_b.is_polarh & params_a.is_acceptor
    )
    sigma[don_acc_pair_mask] = global_params.lj_hbond_hdis  # lj_hbond_hdis

    # lj
    sigma6 = sigma * sigma * sigma
    sigma6 = sigma6 * sigma6
    sigma12 = sigma6 * sigma6
    wdepth = torch.sqrt(params_a.lj_wdepth * params_b.lj_wdepth)

    params_p.lj_sigma[:] = sigma
    params_p.lj_wdepth[:] = wdepth
    params_p.lj_coeff_sigma6[:] = -2 * wdepth * sigma6
    params_p.lj_coeff_sigma12[:] = wdepth * sigma12

    # linear part
    # (slope@switch_point*sigma/wdepth)
    lj_switch_value2wdepth = (
        global_params.lj_switch_dis2sigma ** -12
        - 2 * global_params.lj_switch_dis2sigma ** -6
    )
    lj_switch_slope_sigma2wdepth = (
        -12.0 * global_params.lj_switch_dis2sigma ** -13
        + 12.0 * global_params.lj_switch_dis2sigma ** -7
    )

    params_p.lj_switch_slope[:] = wdepth / sigma * lj_switch_slope_sigma2wdepth
    params_p.lj_switch_intercept[:] = (
        wdepth * lj_switch_value2wdepth
        - params_p.lj_switch_slope * sigma * global_params.lj_switch_dis2sigma
    )

    lbx = global_params.spline_start
    # ubx = global_params.max_dis
    params_p.lj_spline_y0[:] = (params_p.lj_coeff_sigma12 * (lbx ** -12)) + (
        params_p.lj_coeff_sigma6 * (lbx ** -6)
    )
    params_p.lj_spline_dy0[:] = (-12 * params_p.lj_coeff_sigma12 * (lbx ** -13)) - (
        6 * params_p.lj_coeff_sigma6 * (lbx ** -7)
    )

    # lk
    inv_neg2_times_pi_sqrt_pi = -0.089793561062583294
    inv_lambda_1 = 1.0 / (params_a.lk_lambda)
    inv_lambda2_1 = inv_lambda_1 * inv_lambda_1
    params_p.lk_inv_lambda2_1[:] = inv_lambda2_1
    params_p.lk_coeff1[:] = (inv_neg2_times_pi_sqrt_pi * params_a.lk_dgfree) * (
        inv_lambda_1 * params_b.lk_volume
    )

    inv_lambda_2 = 1.0 / (params_b.lk_lambda)
    inv_lambda2_2 = inv_lambda_2 * inv_lambda_2
    params_p.lk_inv_lambda2_2[:] = inv_lambda2_2
    params_p.lk_coeff2[:] = (inv_neg2_times_pi_sqrt_pi * params_b.lk_dgfree) * (
        inv_lambda_2 * params_a.lk_volume
    )

    thresh_dis = global_params.lj_switch_dis2sigma * sigma
    inv_thresh_dis2 = 1.0 / (thresh_dis * thresh_dis)
    dis_rad1 = thresh_dis - params_p.lj_rad1
    x_thresh1 = (dis_rad1 * dis_rad1) * params_p.lk_inv_lambda2_1
    dis_rad2 = thresh_dis - params_p.lj_rad2
    x_thresh2 = (dis_rad2 * dis_rad2) * params_p.lk_inv_lambda2_2

    spline_close1_y0 = torch.exp(-x_thresh1) * params_p.lk_coeff1 * inv_thresh_dis2
    spline_close2_y0 = torch.exp(-x_thresh2) * params_p.lk_coeff2 * inv_thresh_dis2
    params_p.lk_spline_close_y0[:] = spline_close1_y0 + spline_close2_y0

    ##
    # near spline
    # fd: in code this is "rounded" to the nearest gridpoint
    switch = torch.min(spline_close1_y0, spline_close2_y0)
    params_p.lk_spline_close_x0[:] = torch.sqrt(
        torch.max(switch * switch - 1.5, switch.new_zeros(1))
    )
    params_p.lk_spline_close_x1[:] = torch.sqrt(switch * switch + 1.0)

    invdist_close = 1 / (params_p.lk_spline_close_x1)
    invdist2_close = invdist_close * invdist_close

    dis_rad_x1 = params_p.lk_spline_close_x1 - params_p.lj_rad1
    x_x1 = (dis_rad_x1 * dis_rad_x1) * params_p.lk_inv_lambda2_1
    y_1 = torch.exp(-x_x1) * params_p.lk_coeff1 * invdist2_close
    dy_1 = -2 * (dis_rad_x1 * params_p.lk_inv_lambda2_1 + invdist_close) * y_1

    dis_rad_x2 = params_p.lk_spline_close_x1 - params_p.lj_rad2
    x_x2 = (dis_rad_x2 * dis_rad_x2) * params_p.lk_inv_lambda2_2
    y_2 = torch.exp(-x_x2) * params_p.lk_coeff2 * invdist2_close
    dy_2 = -2 * (dis_rad_x2 * params_p.lk_inv_lambda2_2 + invdist_close) * y_2

    params_p.lk_spline_close_y1[:] = y_1 + y_2
    params_p.lk_spline_close_dy1[:] = dy_1 + dy_2

    ##
    # far spline
    invdist_far = 1 / (global_params.spline_start)
    invdist2_far = invdist_far * invdist_far

    dis_rad_x3 = global_params.spline_start - params_p.lj_rad1
    x_x3 = (dis_rad_x3 * dis_rad_x3) * params_p.lk_inv_lambda2_1
    y_3 = torch.exp(-x_x3) * params_p.lk_coeff1 * invdist2_far
    dy_3 = -2 * (dis_rad_x3 * params_p.lk_inv_lambda2_1 + invdist_far) * y_3

    dis_rad_x4 = global_params.spline_start - params_p.lj_rad2
    x_x4 = (dis_rad_x4 * dis_rad_x4) * params_p.lk_inv_lambda2_2
    y_4 = torch.exp(-x_x4) * params_p.lk_coeff2 * invdist2_far
    dy_4 = -2 * (dis_rad_x4 * params_p.lk_inv_lambda2_2 + invdist_far) * y_4

    params_p.lk_spline_far_y0[:] = y_3 + y_4
    params_p.lk_spline_far_dy0[:] = dy_3 + dy_4

    return params_p


@validate_args
def lj_score(
    # Pair conf/bond dependent inputs
    dist: Params,
    bonded_path_length: Tensor(torch.uint8)[...],
    # Pair score parameters
    lj_sigma: Params,
    lj_switch_slope: Params,
    lj_switch_intercept: Params,
    lj_coeff_sigma12: Params,
    lj_coeff_sigma6: Params,
    lj_spline_y0: Params,
    lj_spline_dy0: Params,
    # Global score parameters
    lj_switch_dis2sigma: Params,
    spline_start: Params,
    max_dis: Params,
):
    real = dist.dtype

    invdist2 = 1 / (dist * dist)
    invdist6 = invdist2 * invdist2 * invdist2
    invdist12 = invdist6 * invdist6

    # linear part
    shortrange_dcut = lj_switch_dis2sigma * lj_sigma
    shortrange_selector = dist < shortrange_dcut
    shortrange_component = dist * lj_switch_slope + lj_switch_intercept

    # analytic 12-6 part
    analytic_selector = (dist >= shortrange_dcut) & (dist < spline_start)
    analytic_component = (lj_coeff_sigma12 * invdist12) + (lj_coeff_sigma6 * invdist6)

    # lr spline fade part
    x0 = spline_start
    x1 = max_dis
    spline_fade_selector = (dist >= x0) & (dist < x1)

    x = dist
    y0 = lj_spline_y0
    dy0 = lj_spline_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    spline_fade_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))

    raw_lj = (
        shortrange_component * shortrange_selector.to(real)
        + analytic_component * analytic_selector.to(real)
        + spline_fade_component * spline_fade_selector.to(real)
    )

    interaction_weight = torch.where(
        bonded_path_length == 4,
        raw_lj.new_full((1,), .2, requires_grad=False),
        raw_lj.new_full((1,), 1, requires_grad=False),
    )

    return torch.where(
        ~torch.isnan(lj_sigma) & (bonded_path_length > 3),
        interaction_weight * raw_lj,
        interaction_weight.new_zeros(1, requires_grad=False),
    )


@validate_args
def lk_score(
    # Pair conf/bond dependent inputs
    dist: Params,
    bonded_path_length: Tensor(torch.uint8)[...],
    # Pair score parameters
    lj_rad1: Params,
    lj_rad2: Params,
    lk_coeff1: Params,
    lk_coeff2: Params,
    lk_inv_lambda2_1: Params,
    lk_inv_lambda2_2: Params,
    lk_spline_close_dy1: Params,
    lk_spline_close_x0: Params,
    lk_spline_close_x1: Params,
    lk_spline_close_y0: Params,
    lk_spline_close_y1: Params,
    lk_spline_far_dy0: Params,
    lk_spline_far_y0: Params,
    # Global score parameters
    spline_start: Params,
    max_dis: Params,
):
    real = dist.dtype

    invdist2 = 1 / (dist * dist)

    flat_selector = dist < lk_spline_close_x0
    flat_component = lk_spline_close_y0

    # "near" spline part
    # we sum both spline coeffs together rather than summing splines
    near_spline_selector = (dist >= lk_spline_close_x0) & (dist < lk_spline_close_x1)
    x = dist
    x0 = lk_spline_close_x0
    x1 = lk_spline_close_x1
    y0 = lk_spline_close_y0
    y1 = lk_spline_close_y1
    dy1 = lk_spline_close_dy1
    u0 = (3.0 / (x1 - x0)) * ((y1 - y0) / (x1 - x0))
    u1 = (3.0 / (x1 - x0)) * (dy1 - (y1 - y0) / (x1 - x0))
    near_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))

    # analytic LK part
    analytic_selector = (dist >= lk_spline_close_x1) & (dist < spline_start)

    dis1 = dist - lj_rad1
    dis2 = dist - lj_rad2
    x1 = dis1 * dis1 * lk_inv_lambda2_1
    x2 = dis2 * dis2 * lk_inv_lambda2_2
    analytic_component = invdist2 * (
        torch.exp(-x1) * lk_coeff1 + torch.exp(-x2) * lk_coeff2
    )

    # "far" spline  part
    # we sum both spline coeffs together rather than summing splines
    x0 = spline_start
    x1 = max_dis
    far_spline_selector = (dist >= x0) & (dist < x1)
    x = dist
    y0 = lk_spline_far_y0
    dy0 = lk_spline_far_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    far_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))

    raw_lk = (
        flat_component * flat_selector.to(real)
        + near_spline_component * near_spline_selector.to(real)
        + analytic_component * analytic_selector.to(real)
        + far_spline_component * far_spline_selector.to(real)
    )

    interaction_weight = torch.where(
        bonded_path_length == 4,
        raw_lk.new_full((1,), .2, requires_grad=False),
        raw_lk.new_full((1,), 1, requires_grad=False),
    )

    return torch.where(
        ~torch.isnan(lk_coeff1) & (bonded_path_length > 3),
        interaction_weight * raw_lk,
        raw_lk.new_zeros(1, requires_grad=False),
    )
