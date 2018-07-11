import numba
from numba import cuda


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


@cuda.jit(device=True)
def load_f1f2s(f1f2s, ind):
    return (
        f1f2s[ind, 0],
        f1f2s[ind, 1],
        f1f2s[ind, 2],
        f1f2s[ind, 3],
        f1f2s[ind, 4],
        f1f2s[ind, 5],
    )


@cuda.jit(device=True)
def add_f1f2s(v1, v2):
    return (
        v1[0] + v2[0],
        v1[1] + v2[1],
        v1[2] + v2[2],
        v1[3] + v2[3],
        v1[4] + v2[4],
        v1[5] + v2[5],
    )


@cuda.jit(device=True)
def save_f1f2s(f1f2s, ind, v):
    (
        f1f2s[ind, 0],
        f1f2s[ind, 1],
        f1f2s[ind, 2],
        f1f2s[ind, 3],
        f1f2s[ind, 4],
        f1f2s[ind, 5],
    ) = v


@cuda.jit(device=True)
def zero_f1f2s():
    zero = numba.float64(0.)
    return (zero, zero, zero, zero, zero, zero)


@cuda.jit
def segscan_f1f2s_up_tree(
        f1f2s_ko,
        dsi2ki,
        derivsum_atom_ranges,
        prior_children,
        is_leaf,
):
    shared_f1f2s = cuda.shared.array((512, 6), numba.float64)
    shared_is_leaf = cuda.shared.array((512), numba.int32)
    pos = cuda.grid(1)

    for depth in range(derivsum_atom_ranges.shape[0]):
        start = derivsum_atom_ranges[depth, 0]
        end = derivsum_atom_ranges[depth, 1]
        blocks_for_depth = (end - start - 1) // 512 + 1

        ### Iterate block across depth generation
        carry_f1f2s = zero_f1f2s()
        carry_is_leaf = False
        for ii in range(blocks_for_depth):
            ii_ind = ii * 512 + start + pos
            # Current index in kinematic ordering
            ii_ko = -1

            ### Load shared memory view of f1f2 block in scan order
            if ii_ind < end:
                ii_ko = dsi2ki[ii_ind]

                ### Read node values from global into shared
                myf1f2s = load_f1f2s(f1f2s_ko, ii_ko)
                shared_is_leaf[pos] = is_leaf[ii_ind]
                my_leaf = shared_is_leaf[pos]

                ### Sum all incoming scan values from children into node
                # incoming values set for any node in scan
                for jj in range(prior_children.shape[1]):
                    jj_child = prior_children[ii_ind, jj]
                    if jj_child != -1:
                        child_f1f2s = load_f1f2s(f1f2s_ko, dsi2ki[jj_child])
                        myf1f2s = add_f1f2s(myf1f2s, child_f1f2s)

                ### Sum carry from previous block if node 0 is non-leaf.
                if pos == 0 and not my_leaf:
                    myf1f2s = add_f1f2s(carry_f1f2s, myf1f2s)
                    my_leaf |= carry_is_leaf
                    shared_is_leaf[0] = my_leaf

                save_f1f2s(shared_f1f2s, pos, myf1f2s)

            ### Sync on prepared shared memory block
            cuda.syncthreads()

            ### Perform parallel segmented scan on block
            offset = 1
            for jj in range(9):  #log2(512) == 8
                if pos >= offset and ii_ind < end:
                    prev_f1f2s = load_f1f2s(shared_f1f2s, pos - offset)
                    prev_leaf = shared_is_leaf[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not my_leaf:
                        myf1f2s = add_f1f2s(myf1f2s, prev_f1f2s)
                        my_leaf |= prev_leaf
                        save_f1f2s(shared_f1f2s, pos, myf1f2s)
                        shared_is_leaf[pos] = my_leaf
                offset *= 2
                cuda.syncthreads()

            ### write the block's f1f2s to global memory
            if ii_ind < end:
                save_f1f2s(f1f2s_ko, ii_ko, myf1f2s)

            ### save the carry
            if pos == 0:
                carry_f1f2s = load_f1f2s(shared_f1f2s, 511)
                carry_is_leaf = shared_is_leaf[511]

            cuda.syncthreads()
