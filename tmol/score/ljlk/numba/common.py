import toolz

import numba
from numpy import sqrt

jit = toolz.curry(numba.jit)(nopython=True, nogil=True)


@jit
def connectivity_weight(bonded_path_length):
    if bonded_path_length > 4:
        return 1.0
    elif bonded_path_length == 4:
        return 0.2
    else:
        return 0.0


@jit
def dist(x, y):
    delt0 = x[0] - y[0]
    delt1 = x[1] - y[1]
    delt2 = x[2] - y[2]

    d = sqrt(delt0 * delt0 + delt1 * delt1 + delt2 * delt2)

    return d


@jit
def dist_and_d_dist(x, y):
    delt0 = x[0] - y[0]
    delt1 = x[1] - y[1]
    delt2 = x[2] - y[2]

    d = sqrt(delt0 * delt0 + delt1 * delt1 + delt2 * delt2)
    d_dist_d_xy = (
        (delt0 / d, delt1 / d, delt2 / d),
        (-delt0 / d, -delt1 / d, -delt2 / d),
    )

    return d, d_dist_d_xy


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
