import math

import numba

_lj_potential_signature = (
    numba.float32,  # dist
    numba.uint8,  # bonded_path_length
    numba.float32,  # lj_sigma
    numba.float32,  # lj_switch_slope
    numba.float32,  # lj_switc_intercept
    numba.float32,  # lj_coeff_sigma12
    numba.float32,  # lj_coeff_sigma6
    numba.float32,  # lj_spline_y0
    numba.float32,  # lj_spline_dy0
    numba.float32,  # lj_switch_dis1sigma
    numba.float32,  # spline_start
    numba.float32,  # max_dis
)


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
    lj = 0.0

    if dist > max_dis:
        # Outside of interaction distance
        return 0.0
    elif bonded_path_length < 4:
        # Within bonded distance
        return 0.0
    elif dist > spline_start:
        # lr spline fade

        x0 = spline_start
        x1 = max_dis

        x = dist
        y0 = lj_spline_y0
        dy0 = lj_spline_dy0
        u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
        u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))

        lj = ((x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)) / (
            3 * (x0 - x1)
        )
    elif dist > lj_switch_dis2sigma * lj_sigma:
        # analytic 12-6

        invdist2 = 1 / (dist * dist)
        invdist6 = invdist2 * invdist2 * invdist2
        invdist12 = invdist6 * invdist6

        lj = (lj_coeff_sigma12 * invdist12) + (lj_coeff_sigma6 * invdist6)
    else:
        # linear
        lj = dist * lj_switch_slope + lj_switch_intercept

    if bonded_path_length == 4:
        lj *= 0.2

    return lj


lj_potential_cpu = numba.njit(_lj_potential_signature)(_lj_potential)
lj_potential_cuda = numba.cuda.jit(_lj_potential_signature, device=True)(_lj_potential)

_lj_kernel_signature = (
    # a coords/types
    numba.float32[:, :],
    numba.int64[:],
    # b coords/types
    numba.float32[:, :],
    numba.int64[:],
    # a_b bonded path length
    numba.uint8[:, :],
    # output buffer
    numba.float32[:, :],
    # pair score params
    numba.float32[:, :],
    numba.float32[:, :],
    numba.float32[:, :],
    numba.float32[:, :],
    numba.float32[:, :],
    numba.float32[:, :],
    numba.float32[:, :],
    # global score params
    numba.float32,
    numba.float32,
    numba.float32,
)


def _lj_kernel_cpu(
    a_coords,
    a_types,
    b_coords,
    b_types,
    a_b_bonded_path_length,
    lj_out,
    # Pair score parameters
    lj_sigma: float,
    lj_switch_slope: float,
    lj_switch_intercept: float,
    lj_coeff_sigma12: float,
    lj_coeff_sigma6: float,
    lj_spline_y0: float,
    lj_spline_dy0: float,
    # Global score parameters
    lj_switch_dis2sigma,
    spline_start,
    max_dis,
):
    for i in numba.prange(a_coords.shape[0]):
        at = a_types[i]
        a = a_coords[i]
        if at == -1:
            continue

        for j in numba.prange(b_coords.shape[0]):
            bt = b_types[j]
            b = b_coords[j]
            if bt == -1:
                continue

            delta = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
            dist = math.sqrt(
                delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            )

            lj_out[i, j] = lj_potential_cpu(
                dist,
                a_b_bonded_path_length[i, j],
                lj_sigma[at, bt],
                lj_switch_slope[at, bt],
                lj_switch_intercept[at, bt],
                lj_coeff_sigma12[at, bt],
                lj_coeff_sigma6[at, bt],
                lj_spline_y0[at, bt],
                lj_spline_dy0[at, bt],
                lj_switch_dis2sigma,
                spline_start,
                max_dis,
            )


lj_kernel_serial_cpu = numba.njit(_lj_kernel_signature, parallel=False)(_lj_kernel_cpu)
lj_kernel_parallel_cpu = numba.njit(_lj_kernel_signature, parallel=True)(_lj_kernel_cpu)


@numba.cuda.jit(_lj_kernel_signature)
def lj_kernel_cuda(
    a_coords,
    a_types,
    b_coords,
    b_types,
    a_b_bonded_path_length,
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

    if i >= a_coords.shape[0] or j >= b_coords.shape[0]:
        return

    at = a_types[i]
    a = a_coords[i]

    bt = b_types[j]
    b = b_coords[j]

    if at == -1 or bt == -1:
        return

    delta = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    dist = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])

    lj_out[i, j] = lj_potential_cuda(
        dist,
        a_b_bonded_path_length[i, j],
        lj_sigma[at, bt],
        lj_switch_slope[at, bt],
        lj_switch_intercept[at, bt],
        lj_coeff_sigma12[at, bt],
        lj_coeff_sigma6[at, bt],
        lj_spline_y0[at, bt],
        lj_spline_dy0[at, bt],
        lj_switch_dis2sigma,
        spline_start,
        max_dis,
    )


def lj_kernel(
    a_coords,
    a_types,
    b_coords,
    b_types,
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
    parallel=True,
):
    result = a_coords.new_zeros(a_coords.shape[0], b_coords.shape[0])

    if a_coords.device.type == "cpu":
        (lj_kernel_parallel_cpu if parallel else lj_kernel_serial_cpu)(
            a_coords.numpy(),
            a_types.numpy(),
            b_coords.numpy(),
            b_types.numpy(),
            a_b_bonded_path_length.numpy(),
            result.numpy(),
            # Pair score parameters
            lj_sigma.numpy(),
            lj_switch_slope.numpy(),
            lj_switch_intercept.numpy(),
            lj_coeff_sigma12.numpy(),
            lj_coeff_sigma6.numpy(),
            lj_spline_y0.numpy(),
            lj_spline_dy0.numpy(),
            # Global score parameters
            float(lj_switch_dis2sigma),
            float(spline_start),
            float(max_dis),
        )
    else:
        assert a_coords.device.type == "cuda"
        blocks_per_grid = ((a_coords.shape[0] // 32) + 1, (b_coords.shape[0] // 32) + 1)
        threads_per_block = (32, 32)

        lj_kernel_cuda[blocks_per_grid, threads_per_block](
            a_coords,
            a_types,
            b_coords,
            b_types,
            a_b_bonded_path_length,
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
