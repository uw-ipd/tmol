import math

import numba
from numpy import float32 as f4

BLOCK_SIZE = 8


def _lj_potential(
    dist,
    bonded_path_length,
    # Pair score parameters
    lj_sigma: float,
    lj_switch_slope: float,
    lj_switch_intercept: float,
    lj_coeff_sigma12: float,
    lj_coeff_sigma6: float,
    lj_spline_y0: float,
    lj_spline_dy0: float,
    # Global score parameters
    lj_switch_dis2sigma: float,
    spline_start: float,
    max_dis: float,
) -> float:

    lj = f4(0)

    if dist > max_dis:
        # Outside of interaction distance
        return 0
    elif bonded_path_length < 4:
        # Within bonded distance
        return 0
    elif dist > spline_start:
        # lr spline fade

        x0 = spline_start
        x1 = max_dis

        x = dist
        y0 = lj_spline_y0
        dy0 = lj_spline_dy0
        u0 = (f4(3) / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
        u1 = (f4(3) / (x1 - x0)) * (y0 / (x1 - x0))

        lj = ((x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + f4(3) * y0)) / (
            f4(3) * (x0 - x1)
        )
    elif dist > lj_switch_dis2sigma * lj_sigma:
        # analytic 12-6

        invdist2 = f4(1) / (dist * dist)
        invdist6 = invdist2 * invdist2 * invdist2
        invdist12 = invdist6 * invdist6

        lj = (lj_coeff_sigma12 * invdist12) + (lj_coeff_sigma6 * invdist6)
    else:
        # linear
        lj = dist * lj_switch_slope + lj_switch_intercept

    if bonded_path_length == 4:
        lj *= f4(0.2)

    return lj


lj_potential = numba.njit(_lj_potential)


@numba.njit
def lj_pair_potential(
    a,
    at,
    b,
    bt,
    a_b_bonded_path_length,
    # Pair score parameters
    lj_sigma,
    lj_switch_slope,
    lj_switch_intercept,
    lj_coeff_sigma12,
    lj_coeff_sigma6,
    lj_spline_y0,
    lj_spline_dy0,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
):
    delta = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    dist = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])

    return lj_potential(
        dist,
        a_b_bonded_path_length,
        lj_sigma[at, bt],
        lj_switch_slope[at, bt],
        lj_switch_intercept[at, bt],
        lj_coeff_sigma12[at, bt],
        lj_coeff_sigma6[at, bt],
        lj_spline_y0[at, bt],
        lj_spline_dy0[at, bt],
        lj_switch_dis2sigma[0],
        spline_start[0],
        max_dis[0],
    )


def _lj_intra_kernel_cpu(
    coords,
    types,
    bonded_path_length,
    lj_out,
    # Pair score parameters
    lj_sigma,
    lj_switch_slope,
    lj_switch_intercept,
    lj_coeff_sigma12,
    lj_coeff_sigma6,
    lj_spline_y0,
    lj_spline_dy0,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
):
    nblocks = coords.shape[0] // BLOCK_SIZE

    for b_i in range(nblocks):
        for b_j in range(b_i, nblocks):

            bs_i = b_i * BLOCK_SIZE
            for i in range(bs_i, bs_i + BLOCK_SIZE):
                a = coords[i]
                at = types[i]

                if at == -1:
                    continue

                bs_j = b_j * BLOCK_SIZE
                for j in range(bs_j, bs_j + BLOCK_SIZE):
                    if j <= i:
                        continue

                    b = coords[j]
                    bt = types[j]

                    if bt == -1:
                        continue

                    lj_out[i, j] = lj_pair_potential(
                        a,
                        at,
                        b,
                        bt,
                        bonded_path_length[i, j],
                        # Pair score parameters
                        lj_sigma,
                        lj_switch_slope,
                        lj_switch_intercept,
                        lj_coeff_sigma12,
                        lj_coeff_sigma6,
                        lj_spline_y0,
                        lj_spline_dy0,
                        # Global score parameters
                        lj_switch_dis2sigma,
                        spline_start,
                        max_dis,
                    )


lj_intra_kernel_serial_cpu = numba.njit(parallel=False, nogil=True)(
    _lj_intra_kernel_cpu
)
lj_intra_kernel_parallel_cpu = numba.njit(parallel=True, nogil=True)(
    _lj_intra_kernel_cpu
)


@numba.cuda.jit
def lj_intra_kernel_cuda(
    coords,
    types,
    bonded_path_length,
    lj_out,
    # Pair score parameters
    lj_sigma,
    lj_switch_slope,
    lj_switch_intercept,
    lj_coeff_sigma12,
    lj_coeff_sigma6,
    lj_spline_y0,
    lj_spline_dy0,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
):
    i, j = numba.cuda.grid(2)

    if i >= j:
        return

    if i >= coords.shape[0] or j >= coords.shape[0]:
        return

    a = coords[i]
    at = types[i]
    if at == -1:
        return

    b = coords[j]
    bt = types[j]
    if bt == -1:
        return

    lj_out[i, j] = lj_pair_potential(
        a,
        at,
        b,
        bt,
        bonded_path_length[i, j],
        # Pair score parameters
        lj_sigma,
        lj_switch_slope,
        lj_switch_intercept,
        lj_coeff_sigma12,
        lj_coeff_sigma6,
        lj_spline_y0,
        lj_spline_dy0,
        # Global score parameters
        lj_switch_dis2sigma,
        spline_start,
        max_dis,
    )


def lj_intra_kernel(
    coords,
    atom_types,
    bonded_path_length,
    # Pair score parameters
    lj_sigma,
    lj_switch_slope,
    lj_switch_intercept,
    lj_coeff_sigma12,
    lj_coeff_sigma6,
    lj_spline_y0,
    lj_spline_dy0,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
    parallel=True,
):
    assert coords.shape[0] % BLOCK_SIZE == 0

    result = coords.new_zeros(coords.shape[0], coords.shape[0])

    if coords.device.type == "cpu":
        (lj_intra_kernel_parallel_cpu if parallel else lj_intra_kernel_serial_cpu)(
            coords.__array__(),
            atom_types.__array__(),
            bonded_path_length.__array__(),
            result.__array__(),
            # Pair score parameters
            lj_sigma.__array__(),
            lj_switch_slope.__array__(),
            lj_switch_intercept.__array__(),
            lj_coeff_sigma12.__array__(),
            lj_coeff_sigma6.__array__(),
            lj_spline_y0.__array__(),
            lj_spline_dy0.__array__(),
            # Global score parameters
            lj_switch_dis2sigma.reshape(1).__array__(),
            spline_start.reshape(1).__array__(),
            max_dis.reshape(1).__array__(),
        )
    else:
        assert coords.device.type == "cuda"
        blocks_per_grid = ((coords.shape[0] // 32) + 1, (coords.shape[0] // 32) + 1)
        threads_per_block = (32, 32)

        lj_intra_kernel_cuda[blocks_per_grid, threads_per_block](
            coords,
            atom_types,
            bonded_path_length,
            result,
            # Pair score parameters
            lj_sigma,
            lj_switch_slope,
            lj_switch_intercept,
            lj_coeff_sigma12,
            lj_coeff_sigma6,
            lj_spline_y0,
            lj_spline_dy0,
            # Global score parameters
            lj_switch_dis2sigma,
            spline_start,
            max_dis,
        )

    return result
