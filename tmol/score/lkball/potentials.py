import torch

from tmol.types.functional import validate_args

from tmol.types.torch import Tensor

#from .params import (LKBallGlobalParams, LKBallTypePairParams)

Params = Tensor(torch.float)[...]
CoordArray = Tensor(torch.double)[:, 3]


# get lk energy, 1-sided
# quite similar to LK code...
@validate_args
def get_lk_1way(
        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,
        spline_start: Params,
        max_dis: Params
):
    real = dist.dtype
    invdist2 = 1 / (dist * dist)

    flat_selector = (dist < lk_spline_close_x0)
    flat_component = lk_spline_close_y0

    near_spline_selector = ((dist >= lk_spline_close_x0) &
                            (dist < lk_spline_close_x1))
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
    ) / (3 * (x0 - x1))  # yapf: disable

    # analytic LK part
    analytic_selector = ((dist >= lk_spline_close_x1) & (dist < spline_start))

    dis = dist - lj_rad
    x = dis * dis * lk_inv_lambda2
    analytic_component = invdist2 * ( torch.exp(-x1) * lk_coeff1 )

    x0 = spline_start
    x1 = max_dis
    far_spline_selector = ((dist >= x0) & (dist < x1))
    x = dist
    y0 = lk_spline_far_y0
    dy0 = lk_spline_far_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    far_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    raw_lk = (
        flat_component * flat_selector.to(real) +
        near_spline_component * near_spline_selector.to(real) +
        analytic_component * analytic_selector.to(real) +
        far_spline_component * far_spline_selector.to(real)
    )

    return raw_lk


# build an acceptor water on base atoms with given d/angle/theta
# (donor waters are trivial and built in-line)
@validate_args
def build_acc_waters(
        a: CoordArray, b: CoordArray, b0: CoordArray, d: Params, angle: Params,
        theta: Params
) -> HTArray:
    natoms, = a.shape

    def unit_norm(v):
        return v / torch.norm(v, dim=-1, keepdim=True)

    # a-b-b0 triple to coordinate frame
    Ms = torch.empty([natoms, 3], dtype=a.double, device=a.device)

    xaxis = Ms[:, :3, 0]
    yaxis = Ms[:, :3, 1]
    zaxis = Ms[:, :3, 2]
    center = Ms[:, :3, 3]

    xaxis[:] = unit_norm(a - b)
    zaxis[:] = unit_norm(torch.cross(xaxis, b0 - a))
    yaxis[:] = unit_norm(torch.cross(zaxis, xaxis))
    center[:] = a

    # transform to matrix
    cph = torch.cos(theta)
    sph = torch.sin(theta)
    cth = torch.cos(angle)
    sth = torch.sin(angle)

    Xforms = torch.empty([natoms, 4, 4], dtype=a.double, device=a.device)
    Xforms[:, 0, 0] = cth
    Xforms[:, 0, 1] = -sth
    Xforms[:, 0, 2] = 0
    Xforms[:, 0, 3] = d * cth
    Xforms[:, 1, 0] = cph * sth
    Xforms[:, 1, 1] = cph * cth
    Xforms[:, 1, 2] = -sph
    Xforms[:, 1, 3] = d * cph * sth
    Xforms[:, 2, 0] = sph * sth
    Xforms[:, 2, 1] = sph * cth
    Xforms[:, 2, 2] = cph
    Xforms[:, 2, 3] = d * sph * sth
    Xforms[:, 3, 0] = 0
    Xforms[:, 3, 1] = 0
    Xforms[:, 3, 2] = 0
    Xforms[:, 3, 3] = 1

    waters = torch.matmul(Ms, Xforms)[:, :3, 3]

    return waters


@validate_args
def lkball_score_donor_1way(
        # Input coordinates
        d: CoordArray,
        h: CoordArray,
        desolv: CoordArray,

        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,

        # Global score parameters
        lkb_ramp_width_A2: Params,
        lkb_dist: Params,
        spline_start: Params,
        max_dis: Params,
):
    # 1 fa_sol energy (=lk_ball_iso)
    lkraw = get_lk_1way(
        dist, lj_rad, lk_coeff, lk_inv_lambda2, lk_spline_close_dy1,
        lk_spline_close_x0, lk_spline_close_x1, lk_spline_close_y0,
        lk_spline_close_y1, lk_spline_far_dy0, lk_spline_far_y0, spline_start,
        max_dis
    )

    # 2 virtual waters
    dhn = (d - h)
    dhn = dhn / dhn.norm(dim=-1).unsqueeze(dim=-1)
    virtW = d + lkb_dist * dhn

    # 3 lk_ball
    d2low = (lj_rad - lkb_ramp_width_A2) * (lj_rad - lkb_ramp_width_A2)
    distW = (desolv - virtW)
    d2_delta = (distW * distW).sum(dim=-1) - d2low
    lk_frac = torch.zeros_like(d2_delta)
    lk_frac[d2_delta > lkb_ramp_width_A2] = 1.0
    fade_selector = (d2_delta > 0) & (d2_delta > lkb_ramp_width_A2)
    lk_frac[fade_selector] = d2_delta[fade_selector] / lkb_ramp_width_A2
    lk_frac[fade_selector
            ] = (1 - lk_frac[fade_selector] * lk_frac[fade_selector])
    lk_frac[fade_selector] = lk_frac[fade_selector] * lk_frac[fade_selector]

    lkball_iso = lkraw
    lkball = lkraw * lk_frac


@validate_args
def lkball_score_sp2_acc_1way(
        # Input coordinates
        a: CoordArray,
        b: CoordArray,
        b0: CoordArray,

        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,

        # Global score parameters
        lkb_ramp_width_A2: Params,
        lkb_dist: Params,
        lkb_angle: Params,
        lkb_tors1: Params,
        lkb_tors2: Params,
        multi_water_fade: Params,
        spline_start: Params,
        max_dis: Params,
):
    # 1 fa_sol energy (=lk_ball_iso)
    lkraw = get_lk_1way(
        dist, lj_rad, lk_coeff, lk_inv_lambda2, lk_spline_close_dy1,
        lk_spline_close_x0, lk_spline_close_x1, lk_spline_close_y0,
        lk_spline_close_y1, lk_spline_far_dy0, lk_spline_far_y0, spline_start,
        max_dis
    )

    # 2 virtual waters
    virtW1 = build_acc_waters(a, b0, b, lkb_dist, lkb_angle, lkb_tors1)
    virtW2 = build_acc_waters(a, b0, b, lkb_dist, lkb_angle, lkb_tors2)

    # 3 lk_ball
    d2low = (lj_rad - lkb_ramp_width_A2) * (lj_rad - lkb_ramp_width_A2)
    distW1 = (desolv - virtW1)
    distW2 = (desolv - virtW2)
    d2_delta = -multi_water_fade * torch.log(
        torch.exp(-(distW1 * distW1).sum(dim=-1) / multi_water_fade) +
        torch.exp(-(distW2 * distW2).sum(dim=-1) / multi_water_fade)
    )
    lk_frac = torch.zeros_like(d2_delta)
    lk_frac[d2_delta > lkb_ramp_width_A2] = 1.0
    fade_selector = (d2_delta > 0) & (d2_delta > lkb_ramp_width_A2)
    lk_frac[fade_selector] = d2_delta[fade_selector] / lkb_ramp_width_A2
    lk_frac[fade_selector
            ] = (1 - lk_frac[fade_selector] * lk_frac[fade_selector])
    lk_frac[fade_selector] = lk_frac[fade_selector] * lk_frac[fade_selector]

    lkball_iso = lkraw
    lkball = lkraw * lk_frac


@validate_args
def lkball_score_sp3_acc_1way(
        # Input coordinates
        a: CoordArray,
        b: CoordArray,
        b0: CoordArray,

        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,

        # Global score parameters
        lkb_ramp_width_A2: Params,
        lkb_dist: Params,
        lkb_angle: Params,
        lkb_tors1: Params,
        lkb_tors2: Params,
        multi_water_fade: Params,
        spline_start: Params,
        max_dis: Params,
):
    # identical to sp3 acceptor with different torsion params
    lkball_score_sp2_acc_1way(
        a, b, b0, dist, lj_rad, lk_coeff, lk_inv_lambda2, lk_spline_close_dy1,
        lk_spline_close_x0, lk_spline_close_x1, lk_spline_close_y0,
        lk_spline_close_y1, lk_spline_far_dy0, lk_spline_far_y0,
        lkb_ramp_width_A2, lkb_dist, lkb_angle, lkb_tors1, lkb_tors2,
        multi_water_fade, spline_start, max_dis
    )


@validate_args
def lkball_score_ring_acc_1way(
        # Input coordinates
        a: CoordArray,
        b: CoordArray,
        b0: CoordArray,

        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,

        # Global score parameters
        lkb_ramp_width_A2: Params,
        lkb_dist: Params,
        spline_start: Params,
        max_dis: Params,
):
    # single water, treat it like donor case
    virt_h = 2 * a - 0.5 * (b + b0)

    lkball_score_donor_1way(
        d, virt_h, dist, lj_rad, lk_coeff, lk_inv_lambda2, lk_spline_close_dy1,
        lk_spline_close_x0, lk_spline_close_x1, lk_spline_close_y0,
        lk_spline_close_y1, lk_spline_far_dy0, lk_spline_far_y0,
        lkb_ramp_width_A2, lkb_dist, spline_start, max_dis
    )


# score lkbridge from water/water pair (e.g., donor, donor)
@validate_args
def lkbridge_score_from_water_water(
        a1: CoordArray,
        w11: CoordArray,
        a2: CoordArray,
        w21: CoordArray,

        # Global score parameters
        lkbr_overlap_width_A2: Params,
        lkbr_overlap_gap2: Params,
        lkbr_angle_widthscale: Params
):



