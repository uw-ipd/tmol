import numba
from numba import cuda


@cuda.jit(device=True)
def identity_ht():
    one = numba.float32(1.0)
    zero = numba.float32(0.0)
    return (
        one, zero, zero, zero, zero, one, zero, zero, zero, zero, one, zero
    )


@cuda.jit(device=True)
def ht_load_from_shared(hts, pos):
    v0 = hts[0, pos]
    v1 = hts[1, pos]
    v2 = hts[2, pos]
    v3 = hts[3, pos]
    v4 = hts[4, pos]
    v5 = hts[5, pos]
    v6 = hts[6, pos]
    v7 = hts[7, pos]
    v8 = hts[8, pos]
    v9 = hts[9, pos]
    v10 = hts[10, pos]
    v11 = hts[11, pos]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


@cuda.jit(device=True)
def ht_save_to_shared(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[i, pos] = ht[i]


@cuda.jit(device=True)
def ht_load_from_nx4x4(hts, pos):
    v0 = hts[pos, 0, 0]
    v1 = hts[pos, 0, 1]
    v2 = hts[pos, 0, 2]
    v3 = hts[pos, 0, 3]
    v4 = hts[pos, 1, 0]
    v5 = hts[pos, 1, 1]
    v6 = hts[pos, 1, 2]
    v7 = hts[pos, 1, 3]
    v8 = hts[pos, 2, 0]
    v9 = hts[pos, 2, 1]
    v10 = hts[pos, 2, 2]
    v11 = hts[pos, 2, 3]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


@cuda.jit(device=True)
def ht_save_to_nx4x4(hts, pos, ht):
    for i in range(12):
        hts[pos, i // 4, i % 4] = ht[i]


@cuda.jit(device=True)
def ht_multiply(ht1, ht2):

    r0 = ht1[0] * ht2[0] + ht1[1] * ht2[4] + ht1[2] * ht2[8]
    r1 = ht1[0] * ht2[1] + ht1[1] * ht2[5] + ht1[2] * ht2[9]
    r2 = ht1[0] * ht2[2] + ht1[1] * ht2[6] + ht1[2] * ht2[10]
    r3 = ht1[0] * ht2[3] + ht1[1] * ht2[7] + ht1[2] * ht2[11] + ht1[3]

    r4 = ht1[4] * ht2[0] + ht1[5] * ht2[4] + ht1[6] * ht2[8]
    r5 = ht1[4] * ht2[1] + ht1[5] * ht2[5] + ht1[6] * ht2[9]
    r6 = ht1[4] * ht2[2] + ht1[5] * ht2[6] + ht1[6] * ht2[10]
    r7 = ht1[4] * ht2[3] + ht1[5] * ht2[7] + ht1[6] * ht2[11] + ht1[7]

    r8 = ht1[8] * ht2[0] + ht1[9] * ht2[4] + ht1[10] * ht2[8]
    r9 = ht1[8] * ht2[1] + ht1[9] * ht2[5] + ht1[10] * ht2[9]
    r10 = ht1[8] * ht2[2] + ht1[9] * ht2[6] + ht1[10] * ht2[10]
    r11 = ht1[8] * ht2[3] + ht1[9] * ht2[7] + ht1[10] * ht2[11] + ht1[11]

    return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)


@cuda.jit(device=True)
def ht_multiply_prev_and_store(pos, offset, myht, shared_hts):
    prevht = ht_load_from_shared(shared_hts, pos - offset)

    myht = ht_multiply(prevht, myht)
    ht_save_to_shared(shared_hts, pos, myht)

    return myht


@cuda.jit
def segscan_ht_intervals_one_thread_block(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    # this should be executed as a single thread block with nthreads = 256

    shared_hts = cuda.shared.array((12, 256), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    pos = cuda.grid(1)

    for depth in range(atom_ranges.shape[0]):
        start = atom_ranges[depth, 0]
        end = atom_ranges[depth, 1]

        niters = (end - start - 1) // 256 + 1
        carry_ht = identity_ht()
        carry_is_root = False
        for ii in range(niters):
            ii_ind = ii * 256 + start + pos
            ki = ri2ki[ii_ind]
            #load data into shared memory
            if ii_ind < end:
                for jj in range(12):
                    shared_hts[jj, pos] = hts_ko[ki, jj // 4, jj % 4]
                shared_is_root[pos] = is_root[ii_ind]
                myht = ht_load_from_shared(shared_hts, pos)
                parent = parent_ind[ii_ind]
                htchanged = False
                if parent != -1:
                    parent_ki = ri2ki[parent]
                    parent_ht = ht_load_from_nx4x4(hts_ko, parent_ki)
                    myht = ht_multiply(parent_ht, myht)
                    htchanged = True
                myroot = shared_is_root[pos]
                if pos == 0 and not myroot:
                    myht = ht_multiply(carry_ht, myht)
                    myroot |= carry_is_root
                    shared_is_root[0] = myroot
                    htchanged = True
                if htchanged:
                    ht_save_to_shared(shared_hts, pos, myht)
            cuda.syncthreads()

            # begin segmented scan on this section
            offset = 1
            for jj in range(8):  #log2(256) == 8
                if pos >= offset and ii_ind < end:
                    prev_ht = ht_load_from_shared(shared_hts, pos - offset)
                    prev_root = shared_is_root[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not myroot:
                        myht = ht_multiply(prev_ht, myht)
                        myroot |= prev_root
                        ht_save_to_shared(shared_hts, pos, myht)
                        shared_is_root[pos] = myroot
                offset *= 2
                cuda.syncthreads()

            # write the shared hts to global memory
            if ii_ind < end:
                ht_save_to_nx4x4(hts_ko, ki, myht)

            # save the carry
            if pos == 0:
                carry_ht = ht_load_from_shared(shared_hts, 255)
                carry_is_root = shared_is_root[255]

            cuda.syncthreads()

    #reorder_final_hts_256(hts_ko, hts, ki2ri)


def segscan_hts_gpu(hts_ko, reordering):
    """Perform a series of segmented scan operations on the input homogeneous transforms
    to compute the coordinates (and coordinate frames) of all atoms in the structure.
    This version uses cuda.syncthreads() calls to ensure that there are no data race
    issues. For this reason, it can be safely run on the CPU using numba's CUDA simulator
    (activated by setting the environment variable NUMBA_ENABLE_CUDASIM=1)"""

    ro = reordering
    stream = cuda.stream()

    segscan_ht_intervals_one_thread_block[1, 256, stream](
        hts_ko, ro.ri2ki, ro.is_root_ro, ro.non_subpath_parent_ro,
        ro.refold_atom_ranges
    )
