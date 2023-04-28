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
        * dist**-2
        * exp(-(((dist - lj_radius_i) / lk_lambda_i) ** 2))
    )


@jit
def f_desolv_d_dist(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j):
    return (
        -lk_volume_j
        * lk_dgfree_i
        / (2 * pi ** (3 / 2) * lk_lambda_i)
        * (  # (f * exp(g))' = f' * exp(g) + f g' exp(g)
            -2 * dist**-3 * exp(-((dist - lj_radius_i) ** 2) / lk_lambda_i**2)
            + dist**-2
            * -(2 * dist - 2 * lj_radius_i)
            / lk_lambda_i**2
            * exp(-((dist - lj_radius_i) ** 2) / lk_lambda_i**2)
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
    d_min = lj_sigma_ij * 0.89

    lk_cpoly_close_dmin = numpy.sqrt(d_min * d_min - 1.45)
    lk_cpoly_close_dmax = numpy.sqrt(d_min * d_min + 1.05)

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
    d_min = lj_sigma_ij * 0.89

    lk_cpoly_close_dmin = numpy.sqrt(d_min * d_min - 1.45)
    lk_cpoly_close_dmax = numpy.sqrt(d_min * d_min + 1.05)

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


@jit
def lk_isotropic_intra(
    coords,
    atom_types,
    bonded_path_lengths,
    lj_radius,
    lk_dgfree,
    lk_lambda,
    lk_volume,
    is_donor,
    is_hydroxyl,
    is_polarh,
    is_acceptor,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    nc = coords.shape[0]
    nout = int((nc * (nc - 1)) / 2)

    oinds = numpy.empty((nout, 2), dtype=numpy.int64)
    oval = numpy.empty((nout,), dtype=coords.dtype)

    v = 0
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            ti = atom_types[i]
            tj = atom_types[j]

            val_ij = lk_isotropic(
                dist(coords[i], coords[j]),
                bonded_path_lengths[i, j],
                lj_radius[ti],
                lk_dgfree[ti],
                lk_lambda[ti],
                lk_volume[ti],
                is_donor[ti],
                is_hydroxyl[ti],
                is_polarh[ti],
                is_acceptor[ti],
                lj_radius[tj],
                lk_dgfree[tj],
                lk_lambda[tj],
                lk_volume[tj],
                is_donor[tj],
                is_hydroxyl[tj],
                is_polarh[tj],
                is_acceptor[tj],
                lj_hbond_dis,
                lj_hbond_OH_donor_dis,
                lj_hbond_hdis,
            )

            if val_ij == 0.0:
                continue

            oinds[v, 0] = i
            oinds[v, 1] = j
            oval[v] = val_ij

            v += 1

    return oinds[:v], oval[:v]


@jit
def lk_isotropic_intra_backward(
    inds,
    d_val,
    coords,
    atom_types,
    bonded_path_lengths,
    lj_radius,
    lk_dgfree,
    lk_lambda,
    lk_volume,
    is_donor,
    is_hydroxyl,
    is_polarh,
    is_acceptor,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    oval = numpy.zeros_like(coords)

    for v in range(inds.shape[0]):
        i = inds[v, 0]
        j = inds[v, 1]

        d, (d_d_d_i, d_d_d_j) = dist_and_d_dist(coords[i], coords[j])

        ti = atom_types[i]
        tj = atom_types[j]

        d_val_d_d = d_lk_isotropic_d_dist(
            d,
            bonded_path_lengths[i, j],
            lj_radius[ti],
            lk_dgfree[ti],
            lk_lambda[ti],
            lk_volume[ti],
            is_donor[ti],
            is_hydroxyl[ti],
            is_polarh[ti],
            is_acceptor[ti],
            lj_radius[tj],
            lk_dgfree[tj],
            lk_lambda[tj],
            lk_volume[tj],
            is_donor[tj],
            is_hydroxyl[tj],
            is_polarh[tj],
            is_acceptor[tj],
            lj_hbond_dis,
            lj_hbond_OH_donor_dis,
            lj_hbond_hdis,
        )

        oval[i, 0] += d_d_d_i[0] * d_val_d_d * d_val[v]
        oval[i, 1] += d_d_d_i[1] * d_val_d_d * d_val[v]
        oval[i, 2] += d_d_d_i[2] * d_val_d_d * d_val[v]

        oval[j, 0] += d_d_d_j[0] * d_val_d_d * d_val[v]
        oval[j, 1] += d_d_d_j[1] * d_val_d_d * d_val[v]
        oval[j, 2] += d_d_d_j[2] * d_val_d_d * d_val[v]

    return oval


@jit
def lk_isotropic_inter(
    coords_a,
    atom_types_a,
    coords_b,
    atom_types_b,
    bonded_path_lengths,
    lj_radius,
    lk_dgfree,
    lk_lambda,
    lk_volume,
    is_donor,
    is_hydroxyl,
    is_polarh,
    is_acceptor,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    nout = coords_a.shape[0] * coords_b.shape[0]

    oinds = numpy.empty((nout, 2), dtype=numpy.int64)
    oval = numpy.empty((nout,), dtype=coords_a.dtype)

    v = 0
    for i in range(coords_a.shape[0]):
        for j in range(coords_b.shape[0]):
            ti = atom_types_a[i]
            tj = atom_types_b[j]

            val_ij = lk_isotropic(
                dist(coords_a[i], coords_b[j]),
                bonded_path_lengths[i, j],
                lj_radius[ti],
                lk_dgfree[ti],
                lk_lambda[ti],
                lk_volume[ti],
                is_donor[ti],
                is_hydroxyl[ti],
                is_polarh[ti],
                is_acceptor[ti],
                lj_radius[tj],
                lk_dgfree[tj],
                lk_lambda[tj],
                lk_volume[tj],
                is_donor[tj],
                is_hydroxyl[tj],
                is_polarh[tj],
                is_acceptor[tj],
                lj_hbond_dis,
                lj_hbond_OH_donor_dis,
                lj_hbond_hdis,
            )

            if val_ij == 0.0:
                continue

            oinds[v, 0] = i
            oinds[v, 1] = j
            oval[v] = val_ij

            v += 1

    return oinds[:v], oval[:v]


@jit
def lk_isotropic_inter_backward(
    inds,
    d_val,
    coords_a,
    atom_types_a,
    coords_b,
    atom_types_b,
    bonded_path_lengths,
    lj_radius,
    lk_dgfree,
    lk_lambda,
    lk_volume,
    is_donor,
    is_hydroxyl,
    is_polarh,
    is_acceptor,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    oval_a = numpy.zeros(coords_a.shape, coords_a.dtype)
    oval_b = numpy.zeros(coords_b.shape, coords_b.dtype)

    for v in range(inds.shape[0]):
        i = inds[v, 0]
        j = inds[v, 1]

        d, (d_d_d_i, d_d_d_j) = dist_and_d_dist(coords_a[i], coords_b[j])

        ti = atom_types_a[i]
        tj = atom_types_b[j]

        d_val_d_d = d_lk_isotropic_d_dist(
            d,
            bonded_path_lengths[i, j],
            lj_radius[ti],
            lk_dgfree[ti],
            lk_lambda[ti],
            lk_volume[ti],
            is_donor[ti],
            is_hydroxyl[ti],
            is_polarh[ti],
            is_acceptor[ti],
            lj_radius[tj],
            lk_dgfree[tj],
            lk_lambda[tj],
            lk_volume[tj],
            is_donor[tj],
            is_hydroxyl[tj],
            is_polarh[tj],
            is_acceptor[tj],
            lj_hbond_dis,
            lj_hbond_OH_donor_dis,
            lj_hbond_hdis,
        )

        oval_a[i, 0] += d_d_d_i[0] * d_val_d_d * d_val[v]
        oval_a[i, 1] += d_d_d_i[1] * d_val_d_d * d_val[v]
        oval_a[i, 2] += d_d_d_i[2] * d_val_d_d * d_val[v]

        oval_b[j, 0] += d_d_d_j[0] * d_val_d_d * d_val[v]
        oval_b[j, 1] += d_d_d_j[1] * d_val_d_d * d_val[v]
        oval_b[j, 2] += d_d_d_j[2] * d_val_d_d * d_val[v]

    return oval_a, oval_b
