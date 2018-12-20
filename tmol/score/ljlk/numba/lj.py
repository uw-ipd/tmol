"""Baseline LJ implementation"""

import math
import numba
import numpy
import toolz

import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

from .common import lj_sigma, connectivity_weight, dist, dist_and_d_dist

jit = toolz.curry(numba.jit)(nopython=True, nogil=True)

interpolate_to_zero = jit(cubic_hermite_polynomial.interpolate_to_zero)
interpolate_to_zero_dx = jit(cubic_hermite_polynomial.interpolate_to_zero_dx)


@jit
def f_vdw(dist, sigma, epsilon):
    return epsilon * ((sigma / dist) ** 12 - 2 * (sigma / dist) ** 6)


@jit
def f_vdw_d_dist(dist, sigma, epsilon):
    return epsilon * (12 * sigma ** 6 / dist ** 7 - 12 * sigma ** 12 / dist ** 13)


@jit
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


@jit
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


@jit
def lj_intra(
    coords,
    atom_types,
    bonded_path_lengths,
    lj_radius,
    lj_wdepth,
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
    oval = numpy.empty((nout,), dtype=numpy.float32)

    v = 0
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):

            ti = atom_types[i]
            tj = atom_types[j]

            val_ij = lj(
                dist(coords[i], coords[j]),
                bonded_path_lengths[i, j],
                lj_radius[ti],
                lj_wdepth[ti],
                is_donor[ti],
                is_hydroxyl[ti],
                is_polarh[ti],
                is_acceptor[ti],
                lj_radius[tj],
                lj_wdepth[tj],
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
def lj_intra_backward(
    inds,
    d_val,
    coords,
    atom_types,
    bonded_path_lengths,
    lj_radius,
    lj_wdepth,
    is_donor,
    is_hydroxyl,
    is_polarh,
    is_acceptor,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    oval = numpy.zeros(coords.shape, coords.dtype)

    for v in range(inds.shape[0]):
        i = inds[v, 0]
        j = inds[v, 1]

        d, (d_d_d_i, d_d_d_j) = dist_and_d_dist(coords[i], coords[j])

        ti = atom_types[i]
        tj = atom_types[j]

        d_val_d_d = d_lj_d_dist(
            d,
            bonded_path_lengths[i, j],
            lj_radius[ti],
            lj_wdepth[ti],
            is_donor[ti],
            is_hydroxyl[ti],
            is_polarh[ti],
            is_acceptor[ti],
            lj_radius[tj],
            lj_wdepth[tj],
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
def lj_inter(
    coords_a,
    atom_types_a,
    coords_b,
    atom_types_b,
    bonded_path_lengths,
    lj_radius,
    lj_wdepth,
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

            val_ij = lj(
                dist(coords_a[i], coords_b[j]),
                bonded_path_lengths[i, j],
                lj_radius[ti],
                lj_wdepth[ti],
                is_donor[ti],
                is_hydroxyl[ti],
                is_polarh[ti],
                is_acceptor[ti],
                lj_radius[tj],
                lj_wdepth[tj],
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
def lj_inter_backward(
    inds,
    d_val,
    coords_a,
    atom_types_a,
    coords_b,
    atom_types_b,
    bonded_path_lengths,
    lj_radius,
    lj_wdepth,
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

        d_val_d_d = d_lj_d_dist(
            d,
            bonded_path_lengths[i, j],
            lj_radius[ti],
            lj_wdepth[ti],
            is_donor[ti],
            is_hydroxyl[ti],
            is_polarh[ti],
            is_acceptor[ti],
            lj_radius[tj],
            lj_wdepth[tj],
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
