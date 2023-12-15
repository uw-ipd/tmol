import numpy
from typing import Optional, Union
from tmol.types.array import NDArray

from tmol.types.functional import validate_args


@validate_args
def exclusive_cumsum1d(
    inds: Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]
) -> Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]:
    return numpy.concatenate(
        (
            numpy.array([0], dtype=inds.dtype),
            numpy.cumsum(inds)[:-1],
        ),
        axis=0,
    )


@validate_args
def exclusive_cumsum2d(
    inds: Union[NDArray[numpy.int32][:, :], NDArray[numpy.int64][:, :]]
) -> Union[NDArray[numpy.int32][:, :], NDArray[numpy.int64][:, :]]:
    return numpy.concatenate(
        (
            numpy.zeros((inds.shape[0], 1), dtype=inds.dtype),
            numpy.cumsum(inds, axis=1)[:, :-1],
        ),
        axis=1,
    )


def invert_mapping(
    a_2_b: Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]],
    n_elements_b: Optional[int] = None,
    sentinel: Optional[int] = -1,
):
    Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]
    """Create the inverse mapping, b_2_a, given the input mapping, a_2_b"""
    if n_elements_b is None:
        n_elements_b = numpy.max(a_2_b) + 1

    b_2_a = numpy.full((n_elements_b,), sentinel, dtype=a_2_b.dtype)

    b_2_a[a_2_b] = numpy.arange(a_2_b.shape[0], dtype=a_2_b.dtype)
    return b_2_a
