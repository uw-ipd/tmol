import numpy
from tmol.utility.ndarray.common_operations import invert_mapping


def test_invert_mapping():
    a_2_b = numpy.array([5, 4, 7, 1, 2, 0], dtype=numpy.int32)
    b_2_a = invert_mapping(a_2_b, 8)

    assert b_2_a.dtype == numpy.int32

    b_2_a_gold = numpy.array([5, 3, 4, -1, 1, 0, -1, 2], dtype=numpy.int32)
    numpy.testing.assert_equal(b_2_a_gold, b_2_a)
