import numba
from numba import cuda

from .scan_jit import GenerationalSegmentedScan


@numba.jit(nopython=True)
def finalize_refold_indices(roots, depth_offset, subpath_child_ko, ri2ki,
                            ki2ri):
    count = depth_offset
    for root in roots:
        nextatom = root
        while nextatom != -1:
            ri2ki[count] = nextatom
            ki2ri[nextatom] = count
            nextatom = subpath_child_ko[nextatom]
            count += 1


class HTScan(GenerationalSegmentedScan):
    val_shape = (3, 4)

    @cuda.jit(device=True)
    def zero():
        one = numba.float32(1.0)
        zero = numba.float32(0.0)

        return (
            (one, zero, zero, zero),
            (zero, one, zero, zero),
            (zero, zero, one, zero),
        )

    @cuda.jit(device=True)
    def load(hts, pos):
        """Load affine-compact ht from [i, 3|4, 4] buffer."""
        return (
            (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
            (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
            (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
        )

    @cuda.jit(device=True)
    def save(hts, pos, ht):
        """Save affine-compact ht to [i, 3|4, 4] buffer."""
        (
            (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
            (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
            (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
        ) = ht

    @cuda.jit(device=True)
    def add(ht1, ht2):
        """matmul affine-compact homogenous transforms."""
        # fmt: off
        return ((
            ht1[0][0] * ht2[0][0] + ht1[0][1] * ht2[1][0] + ht1[0][2] * ht2[2][0],
            ht1[0][0] * ht2[0][1] + ht1[0][1] * ht2[1][1] + ht1[0][2] * ht2[2][1],
            ht1[0][0] * ht2[0][2] + ht1[0][1] * ht2[1][2] + ht1[0][2] * ht2[2][2],
            ht1[0][0] * ht2[0][3] + ht1[0][1] * ht2[1][3] + ht1[0][2] * ht2[2][3] + ht1[0][3], # noqa
        ), (
            ht1[1][0] * ht2[0][0] + ht1[1][1] * ht2[1][0] + ht1[1][2] * ht2[2][0],
            ht1[1][0] * ht2[0][1] + ht1[1][1] * ht2[1][1] + ht1[1][2] * ht2[2][1],
            ht1[1][0] * ht2[0][2] + ht1[1][1] * ht2[1][2] + ht1[1][2] * ht2[2][2],
            ht1[1][0] * ht2[0][3] + ht1[1][1] * ht2[1][3] + ht1[1][2] * ht2[2][3] + ht1[1][3], # noqa
        ), (
            ht1[2][0] * ht2[0][0] + ht1[2][1] * ht2[1][0] + ht1[2][2] * ht2[2][0],
            ht1[2][0] * ht2[0][1] + ht1[2][1] * ht2[1][1] + ht1[2][2] * ht2[2][1],
            ht1[2][0] * ht2[0][2] + ht1[2][1] * ht2[1][2] + ht1[2][2] * ht2[2][2],
            ht1[2][0] * ht2[0][3] + ht1[2][1] * ht2[1][3] + ht1[2][2] * ht2[2][3] + ht1[2][3], # noqa
        ))
        # fmt: on
