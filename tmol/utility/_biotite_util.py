import numpy as np
from biotite.structure import get_residue_starts


def get_all_segment_positions(starts, length):
    """Return the segment index of each position from exclusive segment starts."""

    segment_changes = np.zeros(length, dtype=int)
    segment_changes[starts[1:-1]] = 1
    return np.cumsum(segment_changes)


def get_all_residue_positions(array):
    """Return the residue index of each atom in a Biotite atom array."""

    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_all_segment_positions(starts, array.array_length())
