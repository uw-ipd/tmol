"""Baseline implicit desolvation implementation."""

import toolz

import numpy
import numba
from numpy import exp, pi


import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

from .common import connectivity_weight, dist, dist_and_d_dist, lj_sigma

jit = toolz.curry(numba.jit)(nopython=True, nogil=True)

interpolate = jit(cubic_hermite_polynomial.interpolate)
interpolate_dx = jit(cubic_hermite_polynomial.interpolate_dx)
interpolate_to_zero = jit(cubic_hermite_polynomial.interpolate_to_zero)
interpolate_to_zero_dx = jit(cubic_hermite_polynomial.interpolate_to_zero_dx)


@jit
def f_desolv(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j):
    return (
        -lk_volume_j
        * lk_dgfree_i
        / (2 * pi ** (3 / 2) * lk_lambda_i)
        * dist ** -2
        * exp(-((dist - lj_radius_i) / lk_lambda_i) ** 2)
    )


@jit
def f_desolv_d_dist(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j):
    return (
        -lk_volume_j
        * lk_dgfree_i
        / (2 * pi ** (3 / 2) * lk_lambda_i)
        * (  # (f * exp(g))' = f' * exp(g) + f g' exp(g)
            -2 * dist ** -3 * exp(-(dist - lj_radius_i) ** 2 / lk_lambda_i ** 2)
            + dist ** -2
            * -(2 * dist - 2 * lj_radius_i)
            / lk_lambda_i ** 2
            * exp(-(dist - lj_radius_i) ** 2 / lk_lambda_i ** 2)
        )
    )


@jit
def lk_isotropic_pair(
    dist,
    bonded_path_length,
    lj_sigma_ij,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lj_radius_j,
    lk_volume_j,
):
    d_min = lj_sigma_ij * .89

    lk_cpoly_close_dmin = d_min - 0.25
    lk_cpoly_close_dmax = d_min + 0.25

    lk_cpoly_far_dmin = 4.5
    lk_cpoly_far_dmax = 6.0

    weight = connectivity_weight(bonded_path_length)

    if dist < lk_cpoly_close_dmin:
        return weight * f_desolv(
            d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
        )
    elif dist < lk_cpoly_close_dmax:
        return weight * interpolate(
            dist,
            lk_cpoly_close_dmin,
            f_desolv(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
            0,
            lk_cpoly_close_dmax,
            f_desolv(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
        )
    elif dist < lk_cpoly_far_dmin:
        return weight * f_desolv(
            dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
        )
    elif dist < lk_cpoly_far_dmax:
        return weight * interpolate_to_zero(
            dist,
            lk_cpoly_far_dmin,
            f_desolv(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            lk_cpoly_far_dmax,
        )
    else:
        return 0.0


@jit
def d_lk_isotropic_pair_d_dist(
    dist,
    bonded_path_length,
    lj_sigma_ij,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lj_radius_j,
    lk_volume_j,
):
    d_min = lj_sigma_ij * .89

    lk_cpoly_close_dmin = d_min - 0.25
    lk_cpoly_close_dmax = d_min + 0.25

    lk_cpoly_far_dmin = 4.5
    lk_cpoly_far_dmax = 6.0

    weight = connectivity_weight(bonded_path_length)

    if dist < lk_cpoly_close_dmin:
        return 0.0
    elif dist < lk_cpoly_close_dmax:
        return weight * interpolate_dx(
            dist,
            lk_cpoly_close_dmin,
            f_desolv(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
            0,
            lk_cpoly_close_dmax,
            f_desolv(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
        )
    elif dist < lk_cpoly_far_dmin:
        return weight * f_desolv_d_dist(
            dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
        )
    elif dist < lk_cpoly_far_dmax:
        return weight * interpolate_to_zero_dx(
            dist,
            lk_cpoly_far_dmin,
            f_desolv(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            lk_cpoly_far_dmax,
        )
    else:
        return 0.0


@jit
def lk_isotropic(
    dist,
    bonded_path_length,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lk_volume_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lk_dgfree_j,
    lk_lambda_j,
    lk_volume_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    lj_sigma_ij = lj_sigma(
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

    return lk_isotropic_pair(
        dist,
        bonded_path_length,
        lj_sigma_ij,
        lj_radius_i,
        lk_dgfree_i,
        lk_lambda_i,
        lj_radius_j,
        lk_volume_j,
    ) + lk_isotropic_pair(
        dist,
        bonded_path_length,
        lj_sigma_ij,
        lj_radius_j,
        lk_dgfree_j,
        lk_lambda_j,
        lj_radius_i,
        lk_volume_i,
    )


@jit
def d_lk_isotropic_d_dist(
    dist,
    bonded_path_length,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lk_volume_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lk_dgfree_j,
    lk_lambda_j,
    lk_volume_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    lj_sigma_ij = lj_sigma(
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

    return d_lk_isotropic_pair_d_dist(
        dist,
        bonded_path_length,
        lj_sigma_ij,
        lj_radius_i,
        lk_dgfree_i,
        lk_lambda_i,
        lj_radius_j,
        lk_volume_j,
    ) + d_lk_isotropic_pair_d_dist(
        dist,
        bonded_path_length,
        lj_sigma_ij,
        lj_radius_j,
        lk_dgfree_j,
        lk_lambda_j,
        lj_radius_i,
        lk_volume_i,
    )
