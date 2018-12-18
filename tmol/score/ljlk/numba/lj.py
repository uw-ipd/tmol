"""Baseline LJ implementation"""

import math
import numba
import toolz
import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

jit = toolz.curry(numba.jit)(nopython=True)

interpolate_to_zero = jit(cubic_hermite_polynomial.interpolate_to_zero)
interpolate_to_zero_dx = jit(cubic_hermite_polynomial.interpolate_to_zero_dx)


@jit
def f_vdw(dist, sigma, epsilon):
    return epsilon * ((sigma / dist) ** 12 - 2 * (sigma / dist) ** 6)


@jit
def f_vdw_d_dist(dist, sigma, epsilon):
    return epsilon * (12 * sigma ** 6 / dist ** 7 - 12 * sigma ** 12 / dist ** 13)


@jit
def connectivity_weight(bonded_path_length):
    if bonded_path_length > 4:
        return 1.0
    elif bonded_path_length == 4:
        return 0.2
    else:
        return 0.0


@jit
def lj_sigma(
    lj_radius_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    if (is_donor_i and not is_hydroxyl_i and is_acceptor_j) or (
        is_donor_j and not is_hydroxyl_j and is_acceptor_i
    ):
        return lj_hbond_dis
    elif (is_donor_i and is_hydroxyl_i and is_acceptor_j) or (
        is_donor_j and is_hydroxyl_j and is_acceptor_i
    ):
        return lj_hbond_OH_donor_dis
    elif (is_polarh_i and is_acceptor_j) or (is_polarh_j and is_acceptor_i):
        return lj_hbond_hdis
    else:
        return lj_radius_i + lj_radius_j


@numba.jit
def lj(
    dist,
    bonded_path_length,
    lj_radius_i,
    lj_wdepth_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lj_wdepth_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    sigma = lj_sigma(
        lj_radius_i,
        is_donor_i,
        is_hydroxyl_i,
        is_polarh_i,
        is_acceptor_i,
        lj_radius_j,
        is_donor_j,
        is_hydroxyl_j,
        is_polarh_j,
        is_acceptor_j,
        lj_hbond_dis,
        lj_hbond_OH_donor_dis,
        lj_hbond_hdis,
    )

    weight = connectivity_weight(bonded_path_length)

    epsilon = math.sqrt(lj_wdepth_i * lj_wdepth_j)

    d_lin = sigma * 0.6
    lj_cpoly_dmin = 4.5
    lj_cpoly_dmax = 6.0

    if dist < d_lin:
        return weight * (
            f_vdw_d_dist(d_lin, sigma, epsilon) * (dist - d_lin)
            + f_vdw(d_lin, sigma, epsilon)
        )
    elif dist < lj_cpoly_dmin:
        return weight * f_vdw(dist, sigma, epsilon)
    elif dist < lj_cpoly_dmax:
        return weight * interpolate_to_zero(
            dist,
            lj_cpoly_dmin,
            f_vdw(lj_cpoly_dmin, sigma, epsilon),
            f_vdw_d_dist(lj_cpoly_dmin, sigma, epsilon),
            lj_cpoly_dmax,
        )
    else:
        return 0.0


@numba.jit
def d_lj_d_dist(
    dist,
    bonded_path_length,
    lj_radius_i,
    lj_wdepth_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lj_wdepth_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    sigma = lj_sigma(
        lj_radius_i,
        is_donor_i,
        is_hydroxyl_i,
        is_polarh_i,
        is_acceptor_i,
        lj_radius_j,
        is_donor_j,
        is_hydroxyl_j,
        is_polarh_j,
        is_acceptor_j,
        lj_hbond_dis,
        lj_hbond_OH_donor_dis,
        lj_hbond_hdis,
    )

    weight = connectivity_weight(bonded_path_length)

    epsilon = math.sqrt(lj_wdepth_i * lj_wdepth_j)

    d_lin = sigma * 0.6
    lj_cpoly_dmin = 4.5
    lj_cpoly_dmax = 6.0

    if dist < d_lin:
        return weight * f_vdw_d_dist(d_lin, sigma, epsilon)
    elif dist < lj_cpoly_dmin:
        return weight * f_vdw_d_dist(dist, sigma, epsilon)
    elif dist < lj_cpoly_dmax:
        return weight * interpolate_to_zero_dx(
            dist,
            lj_cpoly_dmin,
            f_vdw(lj_cpoly_dmin, sigma, epsilon),
            f_vdw_d_dist(lj_cpoly_dmin, sigma, epsilon),
            lj_cpoly_dmax,
        )
    else:
        return 0.0
