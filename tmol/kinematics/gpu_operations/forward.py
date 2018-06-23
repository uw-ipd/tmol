import numba
from numba import cuda


@cuda.jit
def reorder_starting_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ro[ri, i] = hts_ko[pos, i // 4, i % 4]


@cuda.jit(device=True)
def reorder_starting_hts_256(hts_ko, hts_ro, ki2ri):
    natoms = hts_ko.shape[0]
    pos = cuda.grid(1)
    niters = (natoms - 1) // 256 + 1
    for ii in range(niters):
        which = pos + 256 * ii
        if which < natoms:
            ri = ki2ri[which]
            for i in range(12):
                hts_ro[ri, i] = hts_ko[which, i // 4, i % 4]


@cuda.jit
def reorder_final_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ko[pos, i // 4, i % 4] = hts_ro[ri, i]


@cuda.jit(device=True)
def reorder_final_hts_256(hts_ko, hts_ro, ki2ri):
    natoms = hts_ko.shape[0]
    pos = cuda.grid(1)
    niters = (natoms - 1) // 256 + 1
    for ii in range(niters):
        which = pos + 256 * ii
        if which < natoms:
            ri = ki2ri[which]
            for i in range(12):
                hts_ko[which, i // 4, i % 4] = hts_ro[ri, i]


@cuda.jit(device=True)
def identity_ht():
    one = numba.float32(1.0)
    zero = numba.float32(0.0)
    return (
        one, zero, zero, zero, zero, one, zero, zero, zero, zero, one, zero
    )


@cuda.jit(device=True)
def ht_load(hts, pos):
    v0 = hts[pos, 0]
    v1 = hts[pos, 1]
    v2 = hts[pos, 2]
    v3 = hts[pos, 3]
    v4 = hts[pos, 4]
    v5 = hts[pos, 5]
    v6 = hts[pos, 6]
    v7 = hts[pos, 7]
    v8 = hts[pos, 8]
    v9 = hts[pos, 9]
    v10 = hts[pos, 10]
    v11 = hts[pos, 11]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


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
def ht_save(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[pos, i] = ht[i]


@cuda.jit(device=True)
def ht_save_to_shared(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[i, pos] = ht[i]


@cuda.jit(device=True)
def ht_save_to_n_x_12(hts, pos, ht):
    for i in range(12):
        hts[pos, i] = ht[i]


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


@cuda.jit
def segscan_ht_intervals_one_thread_block(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    # this should be executed as a single thread block with nthreads = 256

    #reorder_starting_hts_256(hts_ko, hts, ki2ri)
    #cuda.syncthreads()

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


@cuda.jit(device=True)
def ht_multiply_prev_and_store(pos, offset, myht, shared_hts):
    prevht = ht_load_from_shared(shared_hts, pos - offset)

    myht = ht_multiply(prevht, myht)
    ht_save_to_shared(shared_hts, pos, myht)

    return myht


@cuda.jit(device=True)
def warp_segscan_hts1(
        pos, warp_id, ri2ki, hts_ko, is_root, parent_ind, carry_ht, shared_hts,
        shared_is_root, int_hts, int_is_root, start, end
):
    ht_ind = start + pos
    lane = pos & 31
    if ht_ind < end:
        shared_is_root[pos] = is_root[ht_ind]

    warp_first = warp_id << 5  # ie warp_id * 32; is this faster than multiplication?
    warp_is_open = shared_is_root[warp_first] == 0

    myht = identity_ht()

    if shared_is_root[pos]:
        shared_is_root[pos] = lane

    # now compute mindex by performing a scan on shared_is_root
    # using "max" as the associative operator
    if lane >= 1:
        shared_is_root[pos] = max(shared_is_root[pos - 1], shared_is_root[pos])
    if lane >= 2:
        shared_is_root[pos] = max(shared_is_root[pos - 2], shared_is_root[pos])
    if lane >= 4:
        shared_is_root[pos] = max(shared_is_root[pos - 4], shared_is_root[pos])
    if lane >= 8:
        shared_is_root[pos] = max(shared_is_root[pos - 8], shared_is_root[pos])
    if lane >= 16:
        shared_is_root[pos] = max(
            shared_is_root[pos - 16], shared_is_root[pos]
        )

    mindex = shared_is_root[pos]
    ki = -1

    # pull down the hts from global memory into shared memory, and then
    # into thread-local memory. Then integrate the parent's HT into root
    # nodes (i.e. nodes whose parent is not listed as -1)
    if ht_ind < end:
        ki = ri2ki[ht_ind]
        for jj in range(12):
            shared_hts[jj, pos] = hts_ko[ki, jj // 4, jj % 4]
        myht = ht_load_from_shared(shared_hts, pos)
        parent = parent_ind[ht_ind]
        htchanged = False
        if parent != -1:
            parent_ki = ri2ki[parent]
            parent_ht = ht_load_from_nx4x4(hts_ko, parent_ki)
            myht = ht_multiply(parent_ht, myht)
            htchanged = True
        if pos == 0 and warp_is_open:
            myht = ht_multiply(carry_ht, myht)
            htchanged = True
        if htchanged:
            ht_save_to_shared(shared_hts, pos, myht)

        # now run segmented scan, unrolling the traditional for loop (does it really save time?)
        # no synchronization necessary for intra-warp scans since these threads are in
        # guaranteed lock sttep
        if lane >= mindex + 1:
            myht = ht_multiply_prev_and_store(pos, 1, myht, shared_hts)
        if lane >= mindex + 2:
            myht = ht_multiply_prev_and_store(pos, 2, myht, shared_hts)
        if lane >= mindex + 4:
            myht = ht_multiply_prev_and_store(pos, 4, myht, shared_hts)
        if lane >= mindex + 8:
            myht = ht_multiply_prev_and_store(pos, 8, myht, shared_hts)
        if lane >= mindex + 16:
            myht = ht_multiply_prev_and_store(pos, 16, myht, shared_hts)

    if lane == 31:
        # now lets write out the intermediate results
        ht_save_to_shared(int_hts, warp_id, myht)
        int_is_root[warp_id] = mindex != 0 or not warp_is_open

    # for the third stage of this intra-block scan, record whether this
    # thread should accumulate the scanned intermediate HT from stage 2
    will_accumulate = warp_is_open and mindex == 0

    return ki, myht, will_accumulate


@cuda.jit(device=True)
def warp_segscan_hts2(pos, int_hts, int_is_root):
    """Now we'll perform a rapid inclusive scan on the int_hts, to merge
    the scanned HTs of all the warps in the thread block. The only threads
    that ought to execute this code are threads 0-7"""

    myht = ht_load_from_shared(int_hts, pos)

    # scan the isroot flags to compute the mindex
    if int_is_root[pos]:
        int_is_root[pos] = pos
    else:
        int_is_root[pos] = 0

    if pos >= 1:
        int_is_root[pos] = max(int_is_root[pos - 1], int_is_root[pos])
    if pos >= 2:
        int_is_root[pos] = max(int_is_root[pos - 2], int_is_root[pos])
    if pos >= 4:
        int_is_root[pos] = max(int_is_root[pos - 4], int_is_root[pos])
    mindex = int_is_root[pos]

    # now scan, unrolling the traditional for loop (does it really save time?)
    if pos >= mindex + 1:
        myht = ht_multiply_prev_and_store(pos, 1, myht, int_hts)
    if pos >= mindex + 2:
        myht = ht_multiply_prev_and_store(pos, 2, myht, int_hts)
    if pos >= mindex + 4:
        myht = ht_multiply_prev_and_store(pos, 4, myht, int_hts)


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
