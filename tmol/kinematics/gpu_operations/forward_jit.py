import math
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
NSCANITER = int(math.log2(NTHREAD))
VAL_SHAPE = (3, 4)
SHARED_SHAPE = (NTHREAD, ) + VAL_SHAPE


@cuda.jit
def segscan_by_generation(
        src_vals,  # [n] + [val_shape]
        scan_to_src_ordering,  # [n]
        is_path_root,  #[n]
        non_path_inputs,  #[n, max_num_inputs]
        generation_ranges,  #[g, 2]
):
    shared_vals = cuda.shared.array(SHARED_SHAPE, numba.float64)
    shared_is_root = cuda.shared.array((NTHREAD), numba.int32)

    pos = cuda.grid(1)

    for gen in range(generation_ranges.shape[0]):
        start = generation_ranges[gen, 0]
        end = generation_ranges[gen, 1]
        blocks_for_gen = (end - start - 1) // NTHREAD + 1

        ### Iterate block across generation
        carry_val = zero()
        carry_is_root = False

        for ii in range(blocks_for_gen):
            ii_ind = ii * NTHREAD + start + pos
            ii_src = -1

            ### Load shared memory value block in scan order
            if ii_ind < end:
                ii_src = scan_to_src_ordering[ii_ind]

                ### Read node values from global into shared
                my_val = load(src_vals, ii_src)
                shared_is_root[pos] = is_path_root[ii_ind]

                ### Sum incoming scan value from parent into node
                # parent only set if node is root of scan
                for jj in range(non_path_inputs.shape[1]):
                    input_ind = non_path_inputs[ii_ind, jj]
                    if input_ind != -1:
                        my_val = add(
                            load(src_vals, scan_to_src_ordering[input_ind]),
                            my_val
                        )

                ### Sum carry value from previous block if node 0 is non-root.
                my_root = shared_is_root[pos]
                if pos == 0 and not my_root:
                    my_val = add(carry_val, my_val)
                    my_root |= carry_is_root
                    shared_is_root[0] = my_root

                save(shared_vals, pos, my_val)

            ### Sync on prepared shared memory block
            cuda.syncthreads()

            ### Perform parallel segmented scan on block
            offset = 1
            for jj in range(NSCANITER):

                if pos >= offset and ii_ind < end:
                    prev_val = load(shared_vals, pos - offset)
                    prev_root = shared_is_root[pos - offset]
                cuda.syncthreads()

                if pos >= offset and ii_ind < end:
                    if not my_root:
                        my_val = add(prev_val, my_val)
                        my_root |= prev_root
                        save(shared_vals, pos, my_val)
                        shared_is_root[pos] = my_root
                offset *= 2
                cuda.syncthreads()

            ### write the block's scan results to global
            if ii_ind < end:
                save(src_vals, ii_src, my_val)

            ### save the carry
            if pos == 0:
                carry_val = load(shared_vals, NTHREAD - 1)
                carry_is_root = shared_is_root[NTHREAD - 1]

            cuda.syncthreads()
