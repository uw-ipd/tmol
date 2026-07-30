import numpy as np
from biotite.structure import get_residue_starts


def get_all_segment_positions(starts, length):
    # backported function from biotite, since it is not available in
    # the version of biotite that is available for python 3.10

    segment_changes = np.zeros(length, dtype=int)
    segment_changes[starts[1:-1]] = 1
    return np.cumsum(segment_changes)


def get_all_residue_positions(array):
    # backported function from biotite, since it is not available in
    # the version of biotite that is available for python 3.10

    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_all_segment_positions(starts, array.array_length())
