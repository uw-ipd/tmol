import toolz
import attr

import pytest
from pytest import approx

import numpy
import scipy.optimize

from tmol.utility.args import ignore_unused_kwargs

import tmol.database

from tmol.database.scoring.ljlk import Hyb

from tmol.score.ljlk.numba.lk_ball import (
    build_acc_waters,
    build_don_water,
    d_don_water_datom,
    d_acc_waters_datom,
    lkball_intra,
)


def test_water_generators(default_database):
    ## test 1: acceptor water generation + derivatives
    a = numpy.array((-6.071, -0.619, -3.193))
    b = numpy.array((-5.250, -1.595, -2.543))
    b0 = numpy.array((-5.489, 0.060, -3.542))
    dist = 2.65
    angle = 109.0
    tors = numpy.array((120.0, 240.0))

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


def test_lk_spotcheck(default_database):
    params = default_database.scoring.ljlk

    # lkball defaults (these should live in DB?)
    water_dist = params.global_parameters.lkb_water_dist
    water_angle_sp2 = params.global_parameters.lkb_water_angle_sp2
    water_angle_sp3 = params.global_parameters.lkb_water_angle_sp3
    water_angle_ring = params.global_parameters.lkb_water_angle_ring
    water_tors_sp2 = params.global_parameters.lkb_water_tors_sp2
    water_tors_sp3 = params.global_parameters.lkb_water_tors_sp3
    water_tors_ring = params.global_parameters.lkb_water_tors_ring
    lj_radius = numpy.array([p.lj_radius for p in params.atom_type_parameters])
    lk_dgfree = numpy.array([p.lk_dgfree for p in params.atom_type_parameters])
    lk_lambda = numpy.array([p.lk_lambda for p in params.atom_type_parameters])
    lk_volume = numpy.array([p.lk_volume for p in params.atom_type_parameters])
    hybridization = numpy.array([p.hybridization for p in params.atom_type_parameters])
    is_donor = numpy.array([p.is_donor for p in params.atom_type_parameters])
    is_hydroxyl = numpy.array([p.is_hydroxyl for p in params.atom_type_parameters])
    is_polarh = numpy.array([p.is_polarh for p in params.atom_type_parameters])
    is_acceptor = numpy.array([p.is_acceptor for p in params.atom_type_parameters])
    lj_hbond_dis = params.global_parameters.lj_hbond_dis
    lj_hbond_OH_donor_dis = params.global_parameters.lj_hbond_dis
    lj_hbond_hdis = params.global_parameters.lj_hbond_dis
    name2index = [p.name for p in params.atom_type_parameters]

    def get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attached_h, base_atoms
    ):
        return lkball_intra(
            coords,
            atom_types,
            bonded_path_lengths,
            attached_h,
            base_atoms,
            lj_radius,
            lk_dgfree,
            lk_lambda,
            lk_volume,
            hybridization,
            is_donor,
            is_hydroxyl,
            is_polarh,
            is_acceptor,
            lj_hbond_dis,
            lj_hbond_OH_donor_dis,
            lj_hbond_hdis,
            water_dist,
            water_angle_sp2,
            water_angle_sp3,
            water_angle_ring,
            water_tors_sp2,
            water_tors_sp3,
            water_tors_ring,
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
    atom_types = numpy.array(
        [
            name2index.index("Nlys"),
            name2index.index("Hpol"),
            name2index.index("Hpol"),
            name2index.index("Hpol"),
            name2index.index("Nlys"),
            name2index.index("Hpol"),
            name2index.index("Hpol"),
            name2index.index("Hpol"),
        ]
    )
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

    # test 2: sp2 acceptor--nonpolar
    coords = numpy.array(
        [
            [-5.282, -0.190, -0.858],
            [-6.520, 0.686, -0.931],
            [-6.652, 1.379, -1.961],
            [-6.932, 4.109, -5.683],
        ]
    )
    atom_types = numpy.array(
        [
            name2index.index("CH2"),
            name2index.index("COO"),
            name2index.index("OOC"),
            name2index.index("CH3"),
        ]
    )

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
    atom_types = numpy.array(
        [
            name2index.index("CH2"),
            name2index.index("OH"),
            name2index.index("Hpol"),
            name2index.index("CH0"),
            name2index.index("NhisDDepro"),
            name2index.index("Caro"),
        ]
    )
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
            [0.09018922, 0.0901892, 0.0265898, 0.4900393],
            [0.00369549, 0.0028072, 0.0, 0.0],
        ]
    )
    numpy.testing.assert_allclose(score, score_ref, atol=1e-5)
