import numpy as np

from tmol_jax.score.dunbrack.dunbrack_energy_term import (
    _fill_missing_rotamer_wells,
)
from tmol_jax.score.dunbrack.potentials import bspline3_coefficients_for_table


def test_periodic_bspline_coefficients_reconstruct_samples():
    samples = np.random.default_rng(7).normal(size=(2, 6, 7)).astype(np.float32)
    coefficients = bspline3_coefficients_for_table(samples, 2)
    reconstructed = coefficients
    for axis in (-2, -1):
        reconstructed = (
            np.roll(reconstructed, 1, axis=axis)
            + 4 * reconstructed
            + np.roll(reconstructed, -1, axis=axis)
        ) / 6
    np.testing.assert_allclose(reconstructed, samples, rtol=2e-6, atol=2e-6)


def test_missing_rotamer_wells_follow_native_neighbour_order():
    probabilities = np.asarray([[0.2], [0.8]], dtype=np.float32)
    mapping = np.asarray([0, -1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int32)

    _fill_missing_rotamer_wells(mapping, probabilities, mirrored=False)

    np.testing.assert_array_equal(mapping, [0, 0, 1, 0, 1, 1, 0, 1, 1])


def test_missing_rotamer_wells_reverse_ties_for_mirrored_tables():
    probabilities = np.asarray([[0.2], [0.8]], dtype=np.float32)
    mapping = np.asarray([0, -1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int32)

    _fill_missing_rotamer_wells(mapping, probabilities, mirrored=True)

    np.testing.assert_array_equal(mapping, [0, 1, 1, 0, 1, 1, 0, 1, 1])


def test_missing_rotamer_wells_fall_back_to_most_probable_row():
    probabilities = np.asarray([[0.2], [0.8]], dtype=np.float32)
    mapping = np.full(9, -1, dtype=np.int32)

    _fill_missing_rotamer_wells(mapping, probabilities, mirrored=False)

    np.testing.assert_array_equal(mapping, np.ones(9, dtype=np.int32))
