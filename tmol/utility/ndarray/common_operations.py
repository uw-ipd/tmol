import numpy
from typing import Optional, Union
from tmol.types.array import NDArray

# from tmol.types.functional import validate_args


# @validate_args
# def exclusive_cumsum1d(
#     inds: Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]
# ) -> Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]] :
#


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


def find_ind_ranges(
    indices: Union[NDArray[numpy.int32], NDArray[numpy.int64]],
    max_ind: Optional[int] = -1,
):
    Union[NDArray[numpy.int32][:], NDArray[numpy.int64][:]]
    """For the input list of (sorted) incides, which may be repeated,
    determine the range for each index for i [b_i, e_i] s.t. for all values
    of b_i <= j < e_i, indices[j] == i. As an upshot, if i is not present
    in indices, then b_i == e_i.
    """
    if max_ind == -1:
        # sorted, therefore highest index in final position
        max_ind = indices[-1] + 1
    int_dtype = indices.dtype
    n_vals = indices.shape[0]
    ranges = numpy.full((max_ind, 2), -1, dtype=indices.dtype)
    diff_from_last = indices[:-1] != indices[1:]
    start_ind_range = numpy.concatenate(
        (numpy.zeros((1,), dtype=int_dtype), numpy.where(diff_from_last)[0] + 1)
    )
    distinct_seen_inds = indices[start_ind_range]

    been_seen = numpy.zeros((max_ind,), dtype=int)
    been_seen[distinct_seen_inds] = 1
    been_seen_cumsum = numpy.cumsum(been_seen) - 1
    reps = numpy.full((max_ind,), -1, dtype=int)
    reps[been_seen_cumsum >= 0] = distinct_seen_inds[
        been_seen_cumsum[been_seen_cumsum >= 0]
    ]

    rep_ranges = numpy.full((max_ind, 2), -1, dtype=indices.dtype)
    rep_ranges[distinct_seen_inds, 0] = start_ind_range
    rep_ranges[distinct_seen_inds[:-1], 1] = start_ind_range[1:]
    rep_ranges[distinct_seen_inds[-1], 1] = n_vals

    ranges[reps != -1, 0] = rep_ranges[reps[reps != -1], 0]
    ranges[reps != -1, 1][:-1] = rep_ranges[reps[reps != -1], 0][1:]
