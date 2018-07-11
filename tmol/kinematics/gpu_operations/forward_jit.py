import numpy
import numba
from numba import cuda


@numba.jit(nopython=True)
def finalize_refold_indices(
        roots, depth_offset, subpath_child_ko, ri2ki, ki2ri
):
    count = depth_offset
    for root in roots:
        nextatom = root
        while nextatom != -1:
            ri2ki[count] = nextatom
            ki2ri[nextatom] = count
            nextatom = subpath_child_ko[nextatom]
            count += 1


@cuda.jit(device=True)
def zero_ht():
    one = numba.float32(1.0)
    zero = numba.float32(0.0)

    return (
        (one, zero, zero, zero),
        (zero, one, zero, zero),
        (zero, zero, one, zero),
    )


@cuda.jit(device=True)
def load_ht(hts, pos):
    """Load affine-compact ht from [i, 3|4, 4] buffer."""
    return (
        (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
        (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
        (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
    )


@cuda.jit(device=True)
def save_ht(hts, pos, ht):
    """Save affine-compact ht to [i, 3|4, 4] buffer."""
    (
        (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
        (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
        (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
    ) = ht


@cuda.jit(device=True)
def add_ht(ht1, ht2):
    """matmul affine-compact homogenous transforms."""
    return ((
        ht1[0][0] * ht2[0][0] + ht1[0][1] * ht2[1][0] + ht1[0][2] * ht2[2][0],
        ht1[0][0] * ht2[0][1] + ht1[0][1] * ht2[1][1] + ht1[0][2] * ht2[2][1],
        ht1[0][0] * ht2[0][2] + ht1[0][1] * ht2[1][2] + ht1[0][2] * ht2[2][2],
        ht1[0][0] * ht2[0][3] + ht1[0][1] * ht2[1][3] + ht1[0][2] * ht2[2][3] +
        ht1[0][3],
    ), (
        ht1[1][0] * ht2[0][0] + ht1[1][1] * ht2[1][0] + ht1[1][2] * ht2[2][0],
        ht1[1][0] * ht2[0][1] + ht1[1][1] * ht2[1][1] + ht1[1][2] * ht2[2][1],
        ht1[1][0] * ht2[0][2] + ht1[1][1] * ht2[1][2] + ht1[1][2] * ht2[2][2],
        ht1[1][0] * ht2[0][3] + ht1[1][1] * ht2[1][3] + ht1[1][2] * ht2[2][3] +
        ht1[1][3],
    ), (
        ht1[2][0] * ht2[0][0] + ht1[2][1] * ht2[1][0] + ht1[2][2] * ht2[2][0],
        ht1[2][0] * ht2[0][1] + ht1[2][1] * ht2[1][1] + ht1[2][2] * ht2[2][1],
        ht1[2][0] * ht2[0][2] + ht1[2][1] * ht2[1][2] + ht1[2][2] * ht2[2][2],
        ht1[2][0] * ht2[0][3] + ht1[2][1] * ht2[1][3] + ht1[2][2] * ht2[2][3] +
        ht1[2][3],
    ))


NTHREAD = 256
NSCANITER = int(numpy.log2(NTHREAD))


@cuda.jit
def segscan_ht_intervals_one_thread_block(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    shared_hts = cuda.shared.array((NTHREAD, 3, 4), numba.float64)
    shared_is_root = cuda.shared.array((NTHREAD), numba.int32)

    pos = cuda.grid(1)

    for depth in range(atom_ranges.shape[0]):
        start = atom_ranges[depth, 0]
        end = atom_ranges[depth, 1]
        blocks_for_depth = (end - start - 1) // NTHREAD + 1

        ### Iterate block across depth generation
        carry_ht = zero_ht()
        carry_is_root = False
        for ii in range(blocks_for_depth):
            ii_ind = ii * NTHREAD + start + pos
            ii_ki = -1

            ### Load shared memory view of HT block in scan order
            if ii_ind < end:
                ii_ki = ri2ki[ii_ind]

                ### Read node values from global into shared
                myht = load_ht(hts_ko, ii_ki)
                shared_is_root[pos] = is_root[ii_ind]
                my_root = shared_is_root[pos]

                ### Sum incoming scan value from parent into node
                # parent only set if node is root of scan
                for jj in range(parent_ind.shape[1]):
                    jj_parent = parent_ind[ii_ind, jj]
                    if jj_parent != -1:
                        myht = add_ht(load_ht(hts_ko, ri2ki[jj_parent]), myht)

                ### Sum carry transform from previous block if node 0 is non-root.
                if pos == 0 and not my_root:
                    myht = add_ht(carry_ht, myht)
                    my_root |= carry_is_root
                    shared_is_root[0] = my_root

                save_ht(shared_hts, pos, myht)

            ### Sync on prepared shared memory block
            cuda.syncthreads()

            ### Perform parallel segmented scan on block
            offset = 1
            for jj in range(NSCANITER):
                if pos >= offset and ii_ind < end:
                    prev_ht = load_ht(shared_hts, pos - offset)
                    prev_root = shared_is_root[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not my_root:
                        myht = add_ht(prev_ht, myht)
                        my_root |= prev_root
                        save_ht(shared_hts, pos, myht)
                        shared_is_root[pos] = my_root
                offset *= 2
                cuda.syncthreads()

            ### write the block's hts to global memory
            if ii_ind < end:
                save_ht(hts_ko, ii_ki, myht)

            ### save the carry
            if pos == 0:
                carry_ht = load_ht(shared_hts, NTHREAD - 1)
                carry_is_root = shared_is_root[NTHREAD - 1]

            cuda.syncthreads()
