import numba
from numba import cuda

from .scan_jit import GenerationalSegmentedScan


@numba.jit(nopython=True)
def finalize_derivsum_indices(
        leaves, start_ind, parent, is_root, ki2dsi, dsi2ki
):
    count = start_ind
    for leaf in leaves:
        nextatom = leaf
        while True:
            dsi2ki[count] = nextatom
            ki2dsi[nextatom] = count
            count += 1
            if is_root[nextatom]:
                break
            nextatom = parent[nextatom]


class F1F2Scan(GenerationalSegmentedScan):
    val_shape = (6, )

    @cuda.jit(device=True)
    def load(f1f2s, ind):
        return (
            f1f2s[ind, 0],
            f1f2s[ind, 1],
            f1f2s[ind, 2],
            f1f2s[ind, 3],
            f1f2s[ind, 4],
            f1f2s[ind, 5],
        )

    @cuda.jit(device=True)
    def add(v1, v2):
        return (
            v1[0] + v2[0],
            v1[1] + v2[1],
            v1[2] + v2[2],
            v1[3] + v2[3],
            v1[4] + v2[4],
            v1[5] + v2[5],
        )

    @cuda.jit(device=True)
    def save(f1f2s, ind, v):
        (
            f1f2s[ind, 0],
            f1f2s[ind, 1],
            f1f2s[ind, 2],
            f1f2s[ind, 3],
            f1f2s[ind, 4],
            f1f2s[ind, 5],
        ) = v

    @cuda.jit(device=True)
    def zero():
        zero = numba.float64(0.)
        return (zero, zero, zero, zero, zero, zero)
