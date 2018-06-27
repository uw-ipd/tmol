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
    v0 = f1f2s[ind, 0]
    v1 = f1f2s[ind, 1]
    v2 = f1f2s[ind, 2]
    v3 = f1f2s[ind, 3]
    v4 = f1f2s[ind, 4]
    v5 = f1f2s[ind, 5]
    return (v0, v1, v2, v3, v4, v5)


@cuda.jit(device=True)
def add_f1f2s(v1, v2):
    res0 = v1[0] + v2[0]
    res1 = v1[1] + v2[1]
    res2 = v1[2] + v2[2]
    res3 = v1[3] + v2[3]
    res4 = v1[4] + v2[4]
    res5 = v1[5] + v2[5]
    return (res0, res1, res2, res3, res4, res5)


@cuda.jit(device=True)
def save_f1f2s(f1f2s, ind, v):
    for i in range(6):
        f1f2s[ind, i] = v[i]


@cuda.jit(device=True)
def zero_f1f2s():
    zero = numba.float64(0.)
    return (zero, zero, zero, zero, zero, zero)


@cuda.jit
def reorder_starting_f1f2s(natoms, f1f2s_ko, f1f2s_dso, ki2dsi):
    pos = cuda.grid(1)
    if pos < natoms:
        dsi = ki2dsi[pos]
        for i in range(6):
            f1f2s_dso[dsi, i] = f1f2s_ko[pos, i]


@cuda.jit
def reorder_final_f1f2s(natoms, f1f2s_ko, f1f2s_dso, ki2dsi):
    pos = cuda.grid(1)
    if pos < natoms:
        dsi = ki2dsi[pos]
        for i in range(6):
            f1f2s_ko[pos, i] = f1f2s_dso[dsi, i]


@cuda.jit
def segscan_f1f2s_up_tree(
        f1f2s_dso, prior_children, is_leaf, derivsum_atom_ranges
):
    shared_f1f2s = cuda.shared.array((512, 6), numba.float64)
    shared_is_leaf = cuda.shared.array((512), numba.int32)
    pos = cuda.grid(1)

    for depth in range(derivsum_atom_ranges.shape[0]):
        start = derivsum_atom_ranges[depth, 0]
        end = derivsum_atom_ranges[depth, 1]

        niters = (end - start - 1) // 512 + 1
        carry_f1f2s = zero_f1f2s()
        carry_is_leaf = False
        for ii in range(niters):
            ii_ind = ii * 512 + start + pos
            if ii_ind < end:
                for jj in range(6):
                    # TO DO: minimize bank conflicts -- align memory reads
                    shared_f1f2s[pos, jj] = f1f2s_dso[ii_ind, jj]
                shared_is_leaf[pos] = is_leaf[ii_ind]
                myf1f2s = load_f1f2s(shared_f1f2s, pos)
                my_leaf = shared_is_leaf[pos]
                f1f2s_changed = False
                for jj in range(prior_children.shape[1]):
                    jj_child = prior_children[ii_ind, jj]
                    if jj_child != -1:
                        child_f1f2s = load_f1f2s(f1f2s_dso, jj_child)
                        myf1f2s = add_f1f2s(myf1f2s, child_f1f2s)
                        f1f2s_changed = True
                if pos == 0 and not my_leaf:
                    myf1f2s = add_f1f2s(carry_f1f2s, myf1f2s)
                    my_leaf |= carry_is_leaf
                    shared_is_leaf[0] = my_leaf
                    f1f2s_changed = True
                if f1f2s_changed:
                    save_f1f2s(shared_f1f2s, pos, myf1f2s)
            cuda.syncthreads()

            # begin segmented scan on this section
            offset = 1
            for jj in range(9):
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

            # write the f1f2s to global memory
            if ii_ind < end:
                save_f1f2s(f1f2s_dso, ii_ind, myf1f2s)

            # save the carry
            if pos == 0:
                carry_f1f2s = load_f1f2s(shared_f1f2s, 511)
                carry_is_leaf = shared_is_leaf[511]

            cuda.syncthreads()
