import toolz
import attr
import pytest
from tmol.utility.units import parse_angle


import torch
import numpy

from tmol.utility.args import ignore_unused_kwargs


from tmol.score.ljlk.params import LJLKParamResolver

from tmol.score.lk_ball.numba.lk_ball import (
    build_acc_waters,
    build_don_water,
    d_don_water_datom,
    d_acc_waters_datom,
    get_lk_fraction,
    get_dlk_fraction_dij,
    get_lkbr_fraction,
    get_dlkbr_fraction_dij,
    lkball_pair,
    dlkball_pair_dij,
    lkball_intra,
    lkball_intra_backward,
)


def test_water_generators(default_database):
    ## test 1: acceptor water generation + derivatives
    a = numpy.array((-6.071, -0.619, -3.193))
    b = numpy.array((-5.250, -1.595, -2.543))
    b0 = numpy.array((-5.489, 0.060, -3.542))
    dist = 2.65
    angle = parse_angle("109.0 deg")
    tors = numpy.array([parse_angle(f"{a} deg") for a in (120.0, 240.0)])

    w = build_acc_waters(a, b, b0, dist, angle, tors)
    waters_ref = numpy.array(
        [[-7.42086525, -1.79165583, -5.14882262], [-7.75428876, 0.40906314, -1.4232189]]
    )
    numpy.testing.assert_allclose(w, waters_ref, atol=1e-6)

    # analytic v numeric derivs
    dWdA = d_acc_waters_datom(a, b, b0, dist, angle, tors)
    gradcheck_delta = 0.0001

    # atom a
    for i in range(3):
        a[i] += gradcheck_delta
        wp = build_acc_waters(a, b, b0, dist, angle, tors)
        a[i] -= 2 * gradcheck_delta
        wm = build_acc_waters(a, b, b0, dist, angle, tors)
        a[i] += gradcheck_delta
        dWdAn = (wp - wm) / (2 * gradcheck_delta)
        numpy.testing.assert_allclose(dWdAn, dWdA[0, i, :, :], atol=1e-5)

    # atom b
    for i in range(3):
        b[i] += gradcheck_delta
        wp = build_acc_waters(a, b, b0, dist, angle, tors)
        b[i] -= 2 * gradcheck_delta
        wm = build_acc_waters(a, b, b0, dist, angle, tors)
        b[i] += gradcheck_delta
        dWdAn = (wp - wm) / (2 * gradcheck_delta)
        numpy.testing.assert_allclose(dWdAn, dWdA[1, i, :, :], atol=1e-5)

    # atom c
    for i in range(3):
        b0[i] += gradcheck_delta
        wp = build_acc_waters(a, b, b0, dist, angle, tors)
        b0[i] -= 2 * gradcheck_delta
        wm = build_acc_waters(a, b, b0, dist, angle, tors)
        b0[i] += gradcheck_delta
        dWdAn = (wp - wm) / (2 * gradcheck_delta)
        numpy.testing.assert_allclose(dWdAn, dWdA[2, i, :, :], atol=1e-5)

    ## test 2: donor water generation + derivatives
    d = numpy.array((-6.007, 4.706, -0.074))
    h = numpy.array((-6.747, 4.361, 0.549))
    dist = 2.65

    w = build_don_water(d, h, dist)
    waters_ref = numpy.array([-7.91642236, 3.81579633, 1.5335272])
    numpy.testing.assert_allclose(w, waters_ref, atol=1e-6)

    # analytic v numeric derivs
    dWdA = d_don_water_datom(d, h, dist)

    # atom d
    for i in range(3):
        d[i] += gradcheck_delta
        wp = build_don_water(d, h, dist)
        d[i] -= 2 * gradcheck_delta
        wm = build_don_water(d, h, dist)
        d[i] += gradcheck_delta
        dWdAn = (wp - wm) / (2 * gradcheck_delta)
        numpy.testing.assert_allclose(dWdAn, dWdA[0, i, :], atol=1e-5)

    # atom d
    for i in range(3):
        h[i] += gradcheck_delta
        wp = build_don_water(d, h, dist)
        h[i] -= 2 * gradcheck_delta
        wm = build_don_water(d, h, dist)
        h[i] += gradcheck_delta
        dWdAn = (wp - wm) / (2 * gradcheck_delta)
        numpy.testing.assert_allclose(dWdAn, dWdA[1, i, :], atol=1e-5)


# test derivatives w.r.t water positions
def test_lkball_deriv(default_database):
    a_j = numpy.array((0.0, 0.0, 0.0))
    b_j = numpy.array((0.0, 0.0, -1.0))
    b0_j = numpy.array((1.0, 0.0, 0.0))
    heavyatom_water_len = 2.65
    angle = parse_angle("109.0 deg")
    tors_j = numpy.array([parse_angle(f"{a} deg") for a in (0.0, 120.0, 240.0)])
    w_j = build_acc_waters(a_j, b_j, b0_j, heavyatom_water_len, angle, tors_j)

    ramp_width_A2 = 3.709
    lj_radius_i = 1.8

    ## TEST 1: LKBALL FRACTION
    a_i = numpy.array((-2.5, 0.1, 2.5))
    lkfrac = get_lk_fraction(a_i, ramp_width_A2, lj_radius_i, w_j)
    assert lkfrac == pytest.approx(.65, abs=.01)

    # analytic
    dlkfrac_i_A, dlkfrac_j_A = get_dlk_fraction_dij(
        a_i, ramp_width_A2, lj_radius_i, w_j
    )

    # numeric
    dlkfrac_i_N = numpy.zeros(3)
    dlkfrac_j_N = numpy.zeros((len(tors_j), 3))
    gradcheck_delta = 0.0001
    for x in range(3):
        a_i[x] += gradcheck_delta
        lkp = get_lk_fraction(a_i, ramp_width_A2, lj_radius_i, w_j)
        a_i[x] -= 2 * gradcheck_delta
        lkm = get_lk_fraction(a_i, ramp_width_A2, lj_radius_i, w_j)
        a_i[x] += gradcheck_delta
        dlkfrac_i_N[x] = (lkp - lkm) / (2 * gradcheck_delta)

        for j in range(len(tors_j)):
            w_j[j, x] += gradcheck_delta
            lkp = get_lk_fraction(a_i, ramp_width_A2, lj_radius_i, w_j)
            w_j[j, x] -= 2 * gradcheck_delta
            lkm = get_lk_fraction(a_i, ramp_width_A2, lj_radius_i, w_j)
            w_j[j, x] += gradcheck_delta
            dlkfrac_j_N[j, x] = (lkp - lkm) / (2 * gradcheck_delta)

    numpy.testing.assert_allclose(dlkfrac_i_N, dlkfrac_i_A, atol=1e-6)
    numpy.testing.assert_allclose(dlkfrac_j_N, dlkfrac_j_A, atol=1e-6)

    ## TEST 2: LKBRIDGE FRACTION
    a_i = numpy.array((0.0, 3.0, 3.0))
    b_i = numpy.array((0.0, 3.0, 4.0))
    b0_i = numpy.array((1.0, 0.0, 0.0))
    tors_i = numpy.array([parse_angle(f"{a} deg") for a in (60.0, 180.0, 300.0)])
    w_i = build_acc_waters(a_i, b_i, b0_i, heavyatom_water_len, angle, tors_i)

    overlap_gap_A2 = 0.5
    overlap_width_A2 = 2.6

    lkbrfrac = get_lkbr_fraction(
        a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
    )
    assert lkbrfrac == pytest.approx(.0265, abs=.001)

    # analytic
    dlkfrac_dai_A, dlkfrac_daj_A, dlkfrac_dwi_A, dlkfrac_dwj_A = get_dlkbr_fraction_dij(
        a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
    )

    # numeric
    dlkfrac_dai_N = numpy.zeros((3))
    dlkfrac_daj_N = numpy.zeros((3))
    dlkfrac_dwi_N = numpy.zeros((len(tors_i), 3))
    dlkfrac_dwj_N = numpy.zeros((len(tors_j), 3))
    gradcheck_delta = 0.0001
    for x in range(3):
        a_i[x] += gradcheck_delta
        lkp = get_lkbr_fraction(
            a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
        )
        a_i[x] -= 2 * gradcheck_delta
        lkm = get_lkbr_fraction(
            a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
        )
        a_i[x] += gradcheck_delta
        dlkfrac_dai_N[x] = (lkp - lkm) / (2 * gradcheck_delta)

        for j in range(len(tors_i)):
            w_i[j, x] += gradcheck_delta
            lkp = get_lkbr_fraction(
                a_i,
                a_j,
                overlap_gap_A2,
                overlap_width_A2,
                w_i,
                w_j,
                heavyatom_water_len,
            )
            w_i[j, x] -= 2 * gradcheck_delta
            lkm = get_lkbr_fraction(
                a_i,
                a_j,
                overlap_gap_A2,
                overlap_width_A2,
                w_i,
                w_j,
                heavyatom_water_len,
            )
            w_i[j, x] += gradcheck_delta
            dlkfrac_dwi_N[j, x] = (lkp - lkm) / (2 * gradcheck_delta)

        a_j[x] += gradcheck_delta
        lkp = get_lkbr_fraction(
            a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
        )
        a_j[x] -= 2 * gradcheck_delta
        lkm = get_lkbr_fraction(
            a_i, a_j, overlap_gap_A2, overlap_width_A2, w_i, w_j, heavyatom_water_len
        )
        a_j[x] += gradcheck_delta
        dlkfrac_daj_N[x] = (lkp - lkm) / (2 * gradcheck_delta)

        for j in range(len(tors_i)):
            w_j[j, x] += gradcheck_delta
            lkp = get_lkbr_fraction(
                a_i,
                a_j,
                overlap_gap_A2,
                overlap_width_A2,
                w_i,
                w_j,
                heavyatom_water_len,
            )
            w_j[j, x] -= 2 * gradcheck_delta
            lkm = get_lkbr_fraction(
                a_i,
                a_j,
                overlap_gap_A2,
                overlap_width_A2,
                w_i,
                w_j,
                heavyatom_water_len,
            )
            w_j[j, x] += gradcheck_delta
            dlkfrac_dwj_N[j, x] = (lkp - lkm) / (2 * gradcheck_delta)

    numpy.testing.assert_allclose(dlkfrac_dai_A, dlkfrac_dai_N, atol=1e-6)
    numpy.testing.assert_allclose(dlkfrac_daj_A, dlkfrac_daj_N, atol=1e-6)
    numpy.testing.assert_allclose(dlkfrac_dwi_A, dlkfrac_dwi_N, atol=1e-6)
    numpy.testing.assert_allclose(dlkfrac_dwj_A, dlkfrac_dwj_N, atol=1e-6)

    ## TEST 3: dlkball_pair_dij ... COMBINING ALL DERIVS
    bonded_path_length = 5
    lj_sigma_ij = 3.1
    lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_i = (1.51, -6.3, 3.5, 10.0)
    lj_radius_j, lk_dgfree_j, lk_lambda_j, lk_volume_j = (1.48, -9.3, 3.5, 12.5)

    def eval_lkball_pair(a_i, w_i, a_j, w_j):
        a_ij = a_i - a_j
        dist_ij = numpy.sqrt(numpy.dot(a_ij, a_ij))
        return numpy.array(
            lkball_pair(
                dist_ij,
                bonded_path_length,
                lj_sigma_ij,
                a_i,
                w_i,
                lj_radius_i,
                lk_dgfree_i,
                lk_lambda_i,
                lk_volume_i,
                a_j,
                w_j,
                lj_radius_j,
                lk_dgfree_j,
                lk_lambda_j,
                lk_volume_j,
                heavyatom_water_len,
            )
        )

    def eval_dlkball_pair(a_i, w_i, a_j, w_j):
        a_ij = a_i - a_j
        dist_ij = numpy.sqrt(numpy.dot(a_ij, a_ij))
        return dlkball_pair_dij(
            dist_ij,
            bonded_path_length,
            lj_sigma_ij,
            a_i,
            w_i,
            lj_radius_i,
            lk_dgfree_i,
            lk_lambda_i,
            lk_volume_i,
            a_j,
            w_j,
            lj_radius_j,
            lk_dgfree_j,
            lk_lambda_j,
            lk_volume_j,
            heavyatom_water_len,
        )

    (dlk_dai_A, dlk_daj_A, dlk_dwi_A, dlk_dwj_A) = eval_dlkball_pair(a_i, w_i, a_j, w_j)

    dlk_dai_N = numpy.zeros((3, 4))
    dlk_daj_N = numpy.zeros((3, 4))
    dlk_dwi_N = numpy.zeros((len(tors_i), 3, 4))
    dlk_dwj_N = numpy.zeros((len(tors_j), 3, 4))
    gradcheck_delta = 0.0001
    for x in range(3):
        a_i[x] += gradcheck_delta
        Ep = eval_lkball_pair(a_i, w_i, a_j, w_j)
        a_i[x] -= 2 * gradcheck_delta
        Em = eval_lkball_pair(a_i, w_i, a_j, w_j)
        a_i[x] += gradcheck_delta
        dlk_dai_N[x, :] = (Ep - Em) / (2 * gradcheck_delta)

        for j in range(len(tors_i)):
            w_i[j, x] += gradcheck_delta
            Ep = eval_lkball_pair(a_i, w_i, a_j, w_j)
            w_i[j, x] -= 2 * gradcheck_delta
            Em = eval_lkball_pair(a_i, w_i, a_j, w_j)
            w_i[j, x] += gradcheck_delta
            dlk_dwi_N[j, x, :] = (Ep - Em) / (2 * gradcheck_delta)

        a_j[x] += gradcheck_delta
        Ep = eval_lkball_pair(a_i, w_i, a_j, w_j)
        a_j[x] -= 2 * gradcheck_delta
        Em = eval_lkball_pair(a_i, w_i, a_j, w_j)
        a_j[x] += gradcheck_delta
        dlk_daj_N[x, :] = (Ep - Em) / (2 * gradcheck_delta)

        for j in range(len(tors_j)):
            w_j[j, x] += gradcheck_delta
            Ep = eval_lkball_pair(a_i, w_i, a_j, w_j)
            w_j[j, x] -= 2 * gradcheck_delta
            Em = eval_lkball_pair(a_i, w_i, a_j, w_j)
            w_j[j, x] += gradcheck_delta
            dlk_dwj_N[j, x, :] = (Ep - Em) / (2 * gradcheck_delta)

    numpy.testing.assert_allclose(dlk_dai_A, dlk_dai_N, atol=1e-6)
    numpy.testing.assert_allclose(dlk_daj_A, dlk_daj_N, atol=1e-6)
    numpy.testing.assert_allclose(dlk_dwi_A, dlk_dwi_N, atol=1e-6)
    numpy.testing.assert_allclose(dlk_dwj_A, dlk_dwj_N, atol=1e-6)


def _get_params(param_resolver: LJLKParamResolver):
    """Pack ljlk param tensors into numpy arrays for numba potential."""

    def _t(t):
        t = t.cpu().numpy()
        if t.ndim == 0:
            return t[()]
        return t

    return toolz.valmap(
        _t,
        toolz.merge(
            attr.asdict(param_resolver.type_params),
            attr.asdict(param_resolver.global_params),
        ),
    )


# full backward pass
def test_lkball_deriv_full_backward_pass(default_database):
    param_resolver: LJLKParamResolver = LJLKParamResolver.from_database(
        default_database.chemical,
        default_database.scoring.ljlk,
        device=torch.device("cpu"),
    )
    params = _get_params(param_resolver)

    def get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        return ignore_unused_kwargs(lkball_intra)(
            coords, atom_types, bonded_path_lengths, attached_h, base_atoms, **params
        )

    def get_dlkball_intra(
        inds, coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        d_val = numpy.ones_like(coords)
        return ignore_unused_kwargs(lkball_intra_backward)(
            inds,
            d_val,
            coords,
            atom_types,
            bonded_path_lengths,
            attached_h,
            base_atoms,
            **params,
        )

    coords = numpy.array(
        [
            [-5.250, -1.595, -2.543],  # SER CB
            [-6.071, -0.619, -3.193],  # SER OG
            [-5.489, 0.060, -3.542],  # SER HG
            [-10.628, 2.294, -1.933],  # HIS CG
            [-9.991, 1.160, -1.435],  # HIS ND1
            [-10.715, 0.960, -0.319],  # HIS CE1
        ]
    )
    atom_types = param_resolver.type_idx(
        ["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"]
    ).numpy()
    bonded_path_lengths = numpy.ones((6, 6), dtype=numpy.int)
    bonded_path_lengths[:3, 3:] = 5
    bonded_path_lengths[3:, :3] = 5
    attachedH = numpy.full((6, 4), -1)
    attachedH[1, 1] = 2
    base_atoms = numpy.full((6, 2), -1)
    base_atoms[1, :] = [0, 2]
    base_atoms[4, :] = [3, 5]
    pairs, score = get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )

    # ANALYTIC
    dE_datm_A = get_dlkball_intra(
        pairs, coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )

    # NUMERIC
    gradcheck_delta = 0.0001
    dE_datm_N = numpy.zeros_like(dE_datm_A)
    for atm in range(coords.shape[0]):
        for x in range(3):
            coords[atm, x] += gradcheck_delta
            _, lkp = get_lkball_intra(
                coords, atom_types, bonded_path_lengths, attachedH, base_atoms
            )
            coords[atm, x] -= 2 * gradcheck_delta
            _, lkm = get_lkball_intra(
                coords, atom_types, bonded_path_lengths, attachedH, base_atoms
            )
            coords[atm, x] += gradcheck_delta
            dE_datm_N[atm, x, :] = (numpy.sum(lkp, axis=0) - numpy.sum(lkm, axis=0)) / (
                2 * gradcheck_delta
            )

    numpy.testing.assert_allclose(dE_datm_A, dE_datm_N, atol=1e-6)


# full forward pass
def test_lkball_spotcheck(default_database):
    param_resolver: LJLKParamResolver = LJLKParamResolver.from_database(
        default_database.chemical,
        default_database.scoring.ljlk,
        device=torch.device("cpu"),
    )

    params = _get_params(param_resolver)

    def get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        return ignore_unused_kwargs(lkball_intra)(
            coords, atom_types, bonded_path_lengths, attached_h, base_atoms, **params
        )

    # test 1: donor--donor
    coords = numpy.array(
        [
            [-6.007, 4.706, -0.074],
            [-6.747, 4.361, 0.549],
            [-5.791, 5.657, 0.240],
            [-6.305, 4.706, -1.040],
            [-10.018, 6.062, -2.221],
            [-9.160, 5.711, -2.665],
            [-9.745, 6.899, -1.697],
            [-10.429, 5.372, -1.610],
        ]
    )

    atom_types = param_resolver.type_idx(
        ["Nlys", "Hpol", "Hpol", "Hpol", "Nlys", "Hpol", "Hpol", "Hpol"]
    ).numpy()
    bonded_path_lengths = numpy.ones((8, 8), dtype=numpy.int)
    bonded_path_lengths[:4, 4:] = 5
    bonded_path_lengths[4:, :4] = 5
    attachedH = numpy.full((8, 4), -1)
    attachedH[0, :3] = numpy.arange(1, 4)
    attachedH[4, :3] = numpy.arange(5, 8)
    base_atoms = numpy.full((8, 2), -1)

    pairs, score = get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )
    pairs_ref = [[0, 4]]
    numpy.testing.assert_array_equal(pairs, pairs_ref)
    score_ref = numpy.array([[0.335514, 0., 0.264926, 0.789612]])
    numpy.testing.assert_allclose(score, score_ref, atol=1e-5)


def test_lkball_spotcheck_sp2_nonpolar(default_database):
    param_resolver: LJLKParamResolver = LJLKParamResolver.from_database(
        default_database.chemical,
        default_database.scoring.ljlk,
        device=torch.device("cpu"),
    )

    params = _get_params(param_resolver)

    def get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        return ignore_unused_kwargs(lkball_intra)(
            coords, atom_types, bonded_path_lengths, attached_h, base_atoms, **params
        )

    # test 2: sp2 acceptor--nonpolar
    coords = numpy.array(
        [
            [-5.282, -0.190, -0.858],
            [-6.520, 0.686, -0.931],
            [-6.652, 1.379, -1.961],
            [-6.932, 4.109, -5.683],
        ]
    )
    atom_types = param_resolver.type_idx(["CH2", "COO", "OOC", "CH3"]).numpy()

    bonded_path_lengths = numpy.ones((4, 4), dtype=numpy.int)
    bonded_path_lengths[:3, 3] = 5
    bonded_path_lengths[3, :3] = 5
    attachedH = numpy.full((4, 4), -1)
    base_atoms = numpy.full((4, 2), -1)
    base_atoms[2, :] = [1, 0]

    pairs, score = get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )
    pairs_ref = [[2, 3]]
    numpy.testing.assert_array_equal(pairs, pairs_ref)
    score_ref = numpy.array([[0.14107985, 0.04765878, 0., 0.]])
    numpy.testing.assert_allclose(score, score_ref, atol=1e-5)


def test_lkball_spotcheck_sp3_ring(default_database):
    param_resolver: LJLKParamResolver = LJLKParamResolver.from_database(
        default_database.chemical,
        default_database.scoring.ljlk,
        device=torch.device("cpu"),
    )

    params = _get_params(param_resolver)

    def get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        return ignore_unused_kwargs(lkball_intra)(
            coords, atom_types, bonded_path_lengths, attached_h, base_atoms, **params
        )

    # test 3: ring acceptor--sp3 acceptor
    coords = numpy.array(
        [
            [-5.250, -1.595, -2.543],  # SER CB
            [-6.071, -0.619, -3.193],  # SER OG
            [-5.489, 0.060, -3.542],  # SER HG
            [-10.628, 2.294, -1.933],  # HIS CG
            [-9.991, 1.160, -1.435],  # HIS ND1
            [-10.715, 0.960, -0.319],  # HIS CE1
        ]
    )
    atom_types = param_resolver.type_idx(
        ["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"]
    ).numpy()
    bonded_path_lengths = numpy.ones((6, 6), dtype=numpy.int)
    bonded_path_lengths[:3, 3:] = 5
    bonded_path_lengths[3:, :3] = 5
    attachedH = numpy.full((6, 4), -1)
    attachedH[1, 1] = 2
    base_atoms = numpy.full((6, 2), -1)
    base_atoms[1, :] = [0, 2]
    base_atoms[4, :] = [3, 5]
    pairs, score = get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )
    pairs_ref = [[0, 4], [1, 3], [1, 4], [1, 5]]
    numpy.testing.assert_array_equal(pairs, pairs_ref)
    score_ref = numpy.array(
        [
            [0.01360676, 0.0135272, 0.0, 0.0],
            [0.00385956, 0.0001626, 0.0, 0.0],
            [0.09018922, 0.0901892, 0.0441963, 0.4900393],
            [0.00369549, 0.0028072, 0.0, 0.0],
        ]
    )
    numpy.testing.assert_allclose(score, score_ref, atol=1e-5)
