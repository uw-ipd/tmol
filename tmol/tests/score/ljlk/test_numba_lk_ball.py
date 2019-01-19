import toolz
import attr

import pytest
from pytest import approx

import numpy
import scipy.optimize

from tmol.utility.args import ignore_unused_kwargs

import tmol.database

from tmol.score.ljlk.numba.lk_ball import (
    build_acc_waters,
    build_don_water,
    d_don_water_datom,
    d_acc_waters_datom,
    Hyb,
    lkball_intra,
)


def test_water_generators(default_database):
    ## test 1: acceptor water generation + derivatives
    a = numpy.array((2.008, 1.323, -0.994))
    b = numpy.array((1.270, 0.922, -0.094))
    b0 = numpy.array((0.103, -0.030, -0.392))
    dist = 2.65
    angle = 109.0
    tors = numpy.array((0.0, 120.0, 240.0))

    w = build_acc_waters(a, b, b0, dist, angle, tors)
    waters_ref = numpy.array(
        [
            [3.84403255, 2.92622689, 0.04578268],
            [0.55741278, 2.64524821, -2.77443714],
            [3.17421022, -0.75936688, -2.14560841],
        ]
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

    # test 1: donor:donor
    name2index = [p.name for p in params.atom_type_parameters]

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
    base_atoms = numpy.full(8, numpy.nan)

    pairs, score = get_lkball_intra(
        coords, atom_types, bonded_path_lengths, attachedH, base_atoms
    )
    pairs_ref = [[0, 4]]
    numpy.testing.assert_array_equal(pairs, pairs_ref)
    score_ref = numpy.array([[0.33551352, 0., 0.02222154, 0.78961248]])
    numpy.testing.assert_allclose(score, score_ref, atol=1e-5)
