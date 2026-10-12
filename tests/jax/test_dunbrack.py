import numpy as np

from tmol_jax.score.dunbrack.dunbrack_energy_term import (
    _fill_missing_rotamer_wells,
)


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
