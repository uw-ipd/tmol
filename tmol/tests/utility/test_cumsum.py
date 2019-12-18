import numpy
import torch

from tmol.utility.cumsum import exclusive_cumsum, exclusive_cumsum1d, exclusive_cumsum2d


def test_numpy_exclusive_cumsum():
    foo = numpy.arange(6, dtype=int) + 3
    foo_cumsum = numpy.cumsum(foo)

    foo_ex_cumsum = exclusive_cumsum(foo)
    assert foo_ex_cumsum.shape == foo.shape

    for i in range(6):
        assert foo_ex_cumsum[i] == (0 if i == 0 else foo_cumsum[i - 1])
