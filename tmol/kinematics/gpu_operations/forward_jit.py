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
def identity_ht():
    one = numba.float32(1.0)
    zero = numba.float32(0.0)

    return (
        (one, zero, zero, zero),
        (zero, one, zero, zero),
        (zero, zero, one, zero),
    )


@cuda.jit(device=True)
def ht_load(hts, pos):
    """Load affine-compact ht from [i, 3|4, 4] buffer."""
    return (
        (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
        (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
        (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
    )


@cuda.jit(device=True)
def ht_save(hts, pos, ht):
    """Save affine-compact ht to [i, 3|4, 4] buffer."""
    (
        (hts[pos, 0, 0], hts[pos, 0, 1], hts[pos, 0, 2], hts[pos, 0, 3]),
        (hts[pos, 1, 0], hts[pos, 1, 1], hts[pos, 1, 2], hts[pos, 1, 3]),
        (hts[pos, 2, 0], hts[pos, 2, 1], hts[pos, 2, 2], hts[pos, 2, 3]),
    ) = ht


@cuda.jit(device=True)
def ht_multiply(ht1, ht2):
    """Matrix-multiply affine-compact homogenous transforms."""
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


@cuda.jit
def segscan_ht_intervals_one_thread_block(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    # this should be executed as a single thread block with nthreads = 256

    shared_hts = cuda.shared.array((256, 3, 4), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    pos = cuda.grid(1)

    for depth in range(atom_ranges.shape[0]):
        start = atom_ranges[depth, 0]
        end = atom_ranges[depth, 1]
        blocks_for_depth = (end - start - 1) // 256 + 1

        ### Iterate block across depth generation
        carry_ht = identity_ht()
        carry_is_root = False
        for ii in range(blocks_for_depth):
            ii_ind = ii * 256 + start + pos
            ii_ki = -1

            ### Load shared memory view of HT block in scan order
            if ii_ind < end:
                ii_ki = ri2ki[ii_ind]

                ### Read node values from global into shared
                myht = ht_load(hts_ko, ii_ki)
                shared_is_root[pos] = is_root[ii_ind]
                my_root = shared_is_root[pos]

                ### Sum incoming scan value from parent into node
                # parent only set if node is root of scan
                parent = parent_ind[ii_ind]

                if parent != -1:
                    parent_ht = ht_load(hts_ko, ri2ki[parent])
                    myht = ht_multiply(parent_ht, myht)

                ### Sum carry transform from previous block if node 0 is non-root.
                if pos == 0 and not my_root:
                    myht = ht_multiply(carry_ht, myht)
                    my_root |= carry_is_root
                    shared_is_root[0] = my_root

                ht_save(shared_hts, pos, myht)

            ### Sync on prepared shared memory block
            cuda.syncthreads()

            ### Perform parallel segmented scan on block
            offset = 1
            for jj in range(8):  #log2(256) == 8
                if pos >= offset and ii_ind < end:
                    prev_ht = ht_load(shared_hts, pos - offset)
                    prev_root = shared_is_root[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not my_root:
                        myht = ht_multiply(prev_ht, myht)
                        my_root |= prev_root
                        ht_save(shared_hts, pos, myht)
                        shared_is_root[pos] = my_root
                offset *= 2
                cuda.syncthreads()

            ### write the block's hts to global memory
            if ii_ind < end:
                ht_save(hts_ko, ii_ki, myht)

            ### save the carry
            if pos == 0:
                carry_ht = ht_load(shared_hts, 255)
                carry_is_root = shared_is_root[255]

            cuda.syncthreads()
