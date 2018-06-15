import torch
import numpy
import numba
from numba import cuda

from tmol.types.functional import validate_args

from .datatypes import KinTree, RefoldData


@validate_args
def refold_data_from_kintree(
        kintree: KinTree, device: torch.device
) -> RefoldData:
    '''Constructs a RefoldData object for a given KinTree

    The RefoldObject divides the tree into a set of paths. Along each
    path is a continuous chain of atoms that either 1) require their
    coordinate frames computed as a cumulative product of homogeneous
    transforms for the coordinate update algorithm, or 2) require the
    cumulative sum of their f1f2 vectors for the derivative calculation
    algorithm. In both cases, these paths can be processed efficiently
    on the GPU using an algorithm called "scan."

    Each path in the tree is labeled with a depth: a path with depth
    i may depend on the values computed for atoms with depths 0..i-1.
    All of the paths of the same depth can be processed in a single
    kernel execution using a variation on scan called "segmented scan."

    In order to divide the tree into these paths, this class constructs
    two reorderings of the atoms: a refold ordering (ro) using refold
    indices (ri) and a derivsum ordering (dso) using derivsum indices (di).
    The original ordering from the kintree is the kintree ordering (ko)
    using kintree indices (ki). The indexing in this code is HAIRY, and
    so all arrays are labled with the indexing they are using. There are
    two sets of interconversion arrays for going from kintree indexing
    to 1) refold indexing, and to 2) derivsum indexing: ki2ri & ri2ki and
    ki2dsi & dsi2ki.

    The algorithm for dividing the tree into paths minimizes the number
    of depths in the tree. For any node in the tree, one of its children
    will be in the same path with it, and all other children will be
    roots of their own subtrees. The minimum-depth division of paths
    is computed by lableling the "branching factor" of each atom. The
    branching factor of an atom is 0 if it has no children; if it does
    have children, it is the largest of the branching factors of what
    the atom designates as its on-path child and one-greater than the
    branching factor of any of its other children. Each node then may
    select the child with the largest branching factor as its on-path
    child to minimize its branching factor. This division produces
    a minimum depth tree of paths.

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.
    '''

    if device.type == "cpu":
        return RefoldData(0, [], [], None, None, None, None, None, None)
    else:
        natoms, ndepths, ri2ki, ki2ri, parent_ko, non_subpath_parent_ro, \
            branching_factor_ko, subpath_child_ko, \
            subpath_length_ko, is_subpath_root_ko, is_subpath_leaf_ko, \
            refold_atom_depth_ko, refold_atom_range_for_depth, subpath_root_ro, \
            is_leaf_dso, n_nonpath_children_ko, \
            derivsum_path_depth_ko, derivsum_atom_range_for_depth, ki2dsi, dsi2ki, \
            non_path_children_ko, non_path_children_dso = \
            construct_refold_and_derivsum_orderings(kintree)

        is_root_ro_d, ki2ri_d, non_subpath_parent_ro_d = \
            send_refold_data_to_gpu(natoms, subpath_root_ro, ri2ki, ki2ri, non_subpath_parent_ro)

        ki2dsi_d, is_leaf_dso_d, non_path_children_dso_d = \
            send_derivsum_data_to_gpu(natoms, ki2dsi, is_leaf_dso, non_path_children_dso)

        return RefoldData(
            natoms, refold_atom_range_for_depth, derivsum_atom_range_for_depth,
            non_subpath_parent_ro_d, is_root_ro_d, ki2ri_d, ki2dsi_d,
            is_leaf_dso_d, non_path_children_dso_d
        )


def construct_refold_and_derivsum_orderings(kintree: KinTree):
    natoms = kintree.id.shape[0]

    # Forward kinematics data initialization

    ri2ki = numpy.full((natoms), -1, dtype="int32")
    ki2ri = numpy.full((natoms), -1, dtype="int32")

    parent_ko = numpy.zeros((natoms), dtype="int32")
    non_subpath_parent_ro = numpy.full((natoms), -1, dtype="int32")
    branching_factor_ko = numpy.full((natoms), -1, dtype="int32")
    subpath_child_ko = numpy.full((natoms), -1, dtype="int32")
    #child_on_refold_subpath_ko = numpy.full((natoms), -1, dtype="int32")
    subpath_length_ko = numpy.zeros((natoms), dtype="int32")
    is_subpath_root_ko = numpy.full((natoms), False, dtype="bool")
    refold_atom_depth_ko = numpy.zeros((natoms), dtype="int32")
    subpath_root_ro = numpy.full((natoms), True, dtype="bool")

    is_subpath_leaf_ko = numpy.full((natoms), False, dtype="bool")
    #derivsum_path_length_ko = numpy.full((natoms), 0, dtype="int32")
    is_leaf_dso = numpy.full((natoms), False, dtype="bool")
    #derivsum_first_child_ko = numpy.full((natoms), -1, dtype="int32")
    n_nonpath_children_ko = numpy.full((natoms), 0, dtype="int32")
    derivsum_path_depth_ko = numpy.full((natoms), -1, dtype="int32")
    derivsum_atom_range_for_depth = []
    ki2dsi = numpy.full((natoms), -1, dtype="int32")
    dsi2ki = numpy.full((natoms), -1, dtype="int32")

    parent_ko[:] = kintree.parent

    compute_branching_factor(
        natoms, parent_ko, branching_factor_ko, subpath_child_ko
    )

    mark_path_children_and_count_nonpath_children(
        natoms, parent_ko, subpath_child_ko, n_nonpath_children_ko,
        is_subpath_root_ko, is_subpath_leaf_ko
    )

    max_n_nonpath_children = max(n_nonpath_children_ko)
    non_path_children_ko = numpy.full((natoms, max_n_nonpath_children),
                                      -1,
                                      dtype="int32")
    non_path_children_dso = numpy.full((natoms, max_n_nonpath_children),
                                       -1,
                                       dtype="int32")

    list_nonpath_children(
        natoms, is_subpath_root_ko, parent_ko, non_path_children_ko
    )

    find_derivsum_path_depths(
        natoms, subpath_child_ko, derivsum_path_depth_ko, non_path_children_ko,
        is_subpath_root_ko, subpath_length_ko
    )

    #subpath_length_ko[:] = derivsum_path_length_ko
    #is_subpath_root_ko[:] = is_subpath_root_ko[:]

    #identify_longest_subpaths(
    #    natoms, parent_ko, subpath_length_ko, subpath_child_ko,
    #    is_subpath_root_ko
    #)

    find_refold_path_depths(
        natoms, parent_ko, refold_atom_depth_ko, is_subpath_root_ko
    )
    ndepths = max(refold_atom_depth_ko) + 1

    refold_atom_range_for_depth = []

    determine_refold_indices(
        natoms, ndepths, refold_atom_depth_ko, is_subpath_root_ko,
        subpath_length_ko, refold_atom_range_for_depth, subpath_child_ko,
        ri2ki, ki2ri, subpath_root_ro, parent_ko, non_subpath_parent_ro
    )

    n_derivsum_depths = derivsum_path_depth_ko[0] + 1

    derivsum_atom_range_for_depth = []
    determine_derivsum_indices(
        natoms, n_derivsum_depths, derivsum_path_depth_ko, subpath_length_ko,
        is_subpath_leaf_ko, is_subpath_root_ko, derivsum_atom_range_for_depth,
        parent_ko, ki2dsi, dsi2ki, non_path_children_ko, non_path_children_dso,
        is_leaf_dso
    )

    #print("ki2dsi"); print(ki2dsi)
    #print("dsi2ki"); print(dsi2ki)
    #print("non_path_children_dso"); print(non_path_children_dso)

    return (
        natoms, ndepths, ri2ki, ki2ri, parent_ko, non_subpath_parent_ro,
        branching_factor_ko, subpath_child_ko, subpath_length_ko,
        is_subpath_root_ko, is_subpath_leaf_ko, refold_atom_depth_ko,
        refold_atom_range_for_depth, subpath_root_ro, is_leaf_dso,
        n_nonpath_children_ko, derivsum_path_depth_ko,
        derivsum_atom_range_for_depth, ki2dsi, dsi2ki, non_path_children_ko,
        non_path_children_dso
    )


@numba.jit(nopython=True)
def compute_branching_factor(
        natoms, parent, branching_factor, branchiest_child
):
    for ii in range(natoms - 1, -1, -1):
        ii_bf = branching_factor[ii]
        if ii_bf == -1:
            ii_bf = 0
            branching_factor[ii] = ii_bf
        ii_parent = parent[ii]
        if ii == ii_parent:
            continue
        parent_bf = branching_factor[ii_parent]
        if parent_bf == -1:
            branching_factor[ii_parent] = ii_bf
            branchiest_child[ii_parent] = ii
        elif ii_bf >= parent_bf:
            branching_factor[ii_parent] = max(ii_bf, parent_bf + 1)
            branchiest_child[ii_parent] = ii


@numba.jit(nopython=True)
def identify_longest_subpaths(
        natoms, parent_ko, subpath_length_ko, subpath_child_ko,
        is_subpath_root_ko
):
    for ii in range(natoms - 1, -1, -1):
        subpath_length_ko[ii] += 1
        ii_subpath = subpath_length_ko[ii]
        ii_parent = parent_ko[ii]
        if subpath_length_ko[ii_parent] < ii_subpath and ii_parent != ii:
            subpath_length_ko[ii_parent] = ii_subpath
            subpath_child_ko[ii_parent] = ii
        is_subpath_root_ko[subpath_child_ko[ii]] = False


@numba.jit(nopython=True)
def find_refold_path_depths(
        natoms, parent_ko, refold_atom_depth_ko, is_subpath_root_ko
):
    for ii in range(natoms):
        ii_parent = parent_ko[ii]
        ii_depth = refold_atom_depth_ko[ii_parent]
        if is_subpath_root_ko[ii] and ii_parent != ii:
            ii_depth += 1
        refold_atom_depth_ko[ii] = ii_depth


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


def determine_refold_indices(
        natoms, ndepths, refold_atom_depth_ko, is_subpath_root_ko,
        subpath_length_ko, refold_atom_range_for_depth, subpath_child_ko,
        ri2ki, ki2ri, subpath_root_ro, parent_ko, non_subpath_parent_ro
):
    # sum the path lengths at each depth
    depth_offsets = numpy.zeros((ndepths), dtype="int32")
    numpy.add.at(
        depth_offsets, refold_atom_depth_ko[is_subpath_root_ko],
        subpath_length_ko[is_subpath_root_ko]
    )
    depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
    depth_offsets[0] = 0
    for i in range(ndepths - 1):
        refold_atom_range_for_depth.append(
            (depth_offsets[i], depth_offsets[i + 1])
        )
    refold_atom_range_for_depth.append((depth_offsets[-1], natoms))

    subpath_roots = numpy.nonzero(is_subpath_root_ko)[0]
    root_depths = refold_atom_depth_ko[subpath_roots]
    for ii in range(ndepths):
        ii_roots = subpath_roots[root_depths == ii]
        finalize_refold_indices(
            ii_roots, depth_offsets[ii], subpath_child_ko, ri2ki, ki2ri
        )

    #print("ri2ki"); print(ri2ki)

    assert numpy.all(ri2ki != -1)
    assert numpy.all(ki2ri != -1)

    subpath_root_ro[:] = is_subpath_root_ko[ri2ki]
    non_subpath_parent_ro[subpath_root_ro] = ki2ri[parent_ko[ri2ki]
                                                   [subpath_root_ro]]
    non_subpath_parent_ro[0] = -1

    #print("non_subpath_parent_ro"); print(non_subpath_parent_ro)


def determine_derivsum_indices(
        natoms, n_derivsum_depths, derivsum_path_depth_ko, subpath_length_ko,
        is_subpath_leaf_ko, is_subpath_root_ko, derivsum_atom_range_for_depth,
        parent_ko, ki2dsi, dsi2ki, non_path_children_ko, non_path_children_dso,
        is_leaf_dso
):

    leaf_path_depths = derivsum_path_depth_ko[is_subpath_leaf_ko]
    leaf_path_lengths = subpath_length_ko[is_subpath_leaf_ko]

    depth_offsets = numpy.zeros((n_derivsum_depths), dtype="int32")
    numpy.add.at(depth_offsets, leaf_path_depths, leaf_path_lengths)
    depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
    depth_offsets[0] = 0

    for ii in range(n_derivsum_depths - 1):
        derivsum_atom_range_for_depth.append(
            (depth_offsets[ii], depth_offsets[ii + 1])
        )
    derivsum_atom_range_for_depth.append((depth_offsets[-1], natoms))
    derivsum_leaves = numpy.nonzero(is_subpath_leaf_ko)[0]
    for ii in range(n_derivsum_depths):
        ii_leaves = derivsum_leaves[leaf_path_depths == ii]
        finalize_derivsum_indices(
            ii_leaves, depth_offsets[ii], parent_ko, is_subpath_root_ko,
            ki2dsi, dsi2ki
        )

    #print("ki2dsi"); print(rd.ki2dsi)
    #print("dsi2ki"); print(rd.dsi2ki)

    assert numpy.all(ki2dsi != -1)
    assert numpy.all(dsi2ki != -1)

    for ii in range(non_path_children_ko.shape[1]):
        child_exists = non_path_children_ko[:, ii] != -1
        non_path_children_dso[child_exists, ii] = ki2dsi[
            non_path_children_ko[child_exists, ii]
        ]
    # now all the identies of the children have been remapped, but they
    # are still in kintree order; so reorder them to derivsum order.
    non_path_children_dso[:] = non_path_children_dso[dsi2ki]
    is_leaf_dso[:] = is_subpath_leaf_ko[dsi2ki]


def send_refold_data_to_gpu(
        natoms, subpath_root_ro, ri2ki, ki2ri, non_subpath_parent_ro
):
    is_root_ro_d = cuda.to_device(subpath_root_ro)
    ki2ri_d = cuda.to_device(ki2ri)
    non_subpath_parent_ro_d = cuda.to_device(non_subpath_parent_ro)

    return is_root_ro_d, ki2ri_d, non_subpath_parent_ro_d


@cuda.jit
def reorder_starting_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ro[ri, i] = hts_ko[pos, i // 4, i % 4]


@cuda.jit
def reorder_final_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ko[pos, i // 4, i % 4] = hts_ro[ri, i]


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
def ht_save(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[pos, i] = ht[i]


@cuda.jit(device=True)
def ht_save_to_n_x_12(hts, pos, ht):
    for i in range(12):
        hts[pos, i] = ht[i]


@cuda.jit(device=True)
def ht_save_to_n_x_4_x_4(hts, pos, ht):
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


#@cuda.jit('float64[:,:], boolean[:], int32[:], int32, int32, int32')
@cuda.jit
def segscan_ht_interval_one_thread_block(
        hts, is_root, parent_ind, natoms, start, end
):
    # this should be executed as a single thread block with nthreads = 256
    # "end" is actually one past the last element; compare ii < end
    shared_hts = cuda.shared.array((256, 12), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    pos = cuda.grid(1)
    niters = (end - start - 1) // 256 + 1
    carry_ht = identity_ht()
    carry_is_root = False
    for ii in range(niters):
        ii_ind = ii * 256 + start + pos
        #load data into shared memory
        if ii_ind < end:
            for jj in range(12):
                # TO DO: minimize bank conflicts -- align memory reads
                shared_hts[pos, jj] = hts[ii_ind, jj]
            shared_is_root[pos] = is_root[ii_ind]
            myht = ht_load(shared_hts, pos)
            parent = parent_ind[ii_ind].item()
            htchanged = False
            if parent != -1:
                parent_ht = ht_load(hts, parent)
                myht = ht_multiply(parent_ht, myht)
                htchanged = True
            myroot = shared_is_root[pos]
            if pos == 0 and not myroot:
                myht = ht_multiply(carry_ht, myht)
                myroot |= carry_is_root
                shared_is_root[0] = myroot
                htchanged = True
            if htchanged:
                ht_save(shared_hts, pos, myht)
        cuda.syncthreads()

        # begin segmented scan on this section
        offset = 1
        for jj in range(8):  #log2(256) == 8
            if pos >= offset and ii_ind < end:
                prev_ht = ht_load(shared_hts, pos - offset)
                prev_root = shared_is_root[pos - offset]
            cuda.syncthreads()
            if pos >= offset and ii_ind < end:
                if not myroot:
                    myht = ht_multiply(prev_ht, myht)
                    myroot |= prev_root
                    ht_save(shared_hts, pos, myht)
                    shared_is_root[pos] = myroot
            offset *= 2
            cuda.syncthreads()

        # write the shared hts to global memory
        if ii_ind < end:
            for jj in range(12):
                hts[ii_ind, jj] = shared_hts[pos, jj]

        # save the carry
        if pos == 0:
            carry_ht = ht_load(shared_hts, 255)
            carry_is_root = shared_is_root[255]

        cuda.syncthreads()


#@cuda.jit(
#    'float64[:,:], float64[:,:], boolean[:], boolean[:], int32[:], int32, int32, int32'
#)
@cuda.jit
def segscan_ht_interval_many_thread_blocks_1(
        hts, hts_int, is_root, is_root_int, parent_ind, natoms, start, end
):
    '''This should be executed as many separate thread blocks with nthreads = 32
    As all 32 threads will run in a single warp, there is no need for synchronization
    steps, as the threads are all executed in lock step.
    "end" is actually one past the last element; compare ii < end'''

    shared_hts = cuda.shared.array((32, 12), numba.float64)
    shared_is_root = cuda.shared.array((32), numba.int32)

    pos = cuda.grid(1)
    warp_pos = pos & 31

    ii_ind = start + pos
    myht = identity_ht()
    myroot = False
    if ii_ind < end:
        for jj in range(12):
            # TO DO: minimize bank conflicts -- align memory reads
            shared_hts[warp_pos, jj] = hts[ii_ind, jj]
        shared_is_root[warp_pos] = is_root[ii_ind]
        myht = ht_load(shared_hts, warp_pos)
        parent = parent_ind[ii_ind].item()
        htchanged = False
        if parent != -1:
            parent_ht = ht_load(hts, parent)
            myht = ht_multiply(parent_ht, myht)
            htchanged = True
        myroot = shared_is_root[warp_pos]
        if htchanged:
            ht_save(shared_hts, warp_pos, myht)
    cuda.syncthreads()

    # begin segmented scan
    offset = 1
    for jj in range(6):  #log2(256) == 8
        if warp_pos >= offset and ii_ind < end:
            if not myroot:
                prev_ht = ht_load(shared_hts, warp_pos - offset)
                prev_root = shared_is_root[warp_pos - offset]
        cuda.syncthreads()
        if warp_pos >= offset and ii_ind < end:
            if not myroot:
                myht = ht_multiply(prev_ht, myht)
                myroot |= prev_root
                ht_save(shared_hts, warp_pos, myht)
                shared_is_root[warp_pos] = myroot
        cuda.syncthreads()
        offset *= 2

    # save the carry
    if warp_pos == 31:
        ht_save(hts_int, cuda.blockIdx.x, myht)
        is_root_int[cuda.blockIdx.x] = myroot


#@cuda.jit('float64[:,:], boolean[:], int32')
@cuda.jit
def segscan_ht_interval_one_thread_block_2(hts, is_root, n_intermediates):
    # this should be executed as a single thread block with nthreads = 256
    # compare ii < natoms
    # There is significant thread synchronization here as there are multiple
    # warps running within this single thread block.
    # 256 is the maximum number of double-precision HTs that fit in shared memory
    # on the GPU that I'm running on, so that's the size thread block
    # that I'll be running.

    shared_hts = cuda.shared.array((256, 12), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    pos = cuda.grid(1)
    n_iterations = (n_intermediates - 1) // 256 + 1
    carry_ht = identity_ht()
    carry_is_root = False
    for ii in range(n_iterations):
        ii_ind = ii * 256 + pos
        #load data into shared memory
        if ii_ind < n_intermediates:
            for jj in range(12):
                # TO DO: minimize bank conflicts -- align memory reads
                shared_hts[pos, jj] = hts[ii_ind, jj]
            shared_is_root[pos] = is_root[ii_ind]
            myht = ht_load(shared_hts, pos)
            myroot = shared_is_root[pos]
            if pos == 0 and not myroot:
                myht = ht_multiply(carry_ht, myht)
                myroot |= carry_is_root
                shared_is_root[0] = myroot
                ht_save(shared_hts, pos, myht)
        cuda.syncthreads()

        # begin segmented scan on this section
        offset = 1
        for jj in range(8):  # log2(256) == 8
            if pos >= offset and ii_ind < n_intermediates:
                prev_ht = ht_load(shared_hts, pos - offset)
                prev_root = shared_is_root[pos - offset]
            cuda.syncthreads()
            if pos >= offset and ii_ind < n_intermediates:
                if not myroot:
                    myht = ht_multiply(prev_ht, myht)
                    myroot |= prev_root
                    ht_save(shared_hts, pos, myht)
                    shared_is_root[pos] = myroot
            offset *= 2
            cuda.syncthreads()

        # write the shared hts to global memory
        if ii_ind < n_intermediates:
            for jj in range(12):
                hts[ii_ind, jj] = shared_hts[pos, jj]

        # save the carry
        if pos == 0:
            carry_ht = ht_load(shared_hts, 255)
            carry_is_root = shared_is_root[255]

        cuda.syncthreads()


#@cuda.jit(
#    'float64[:,:], float64[:,:], boolean[:], int32[:], int32, int32, int32'
#)
@cuda.jit
def segscan_ht_interval_many_thread_blocks_3(
        hts, hts_int, is_root, parent_ind, natoms, start, end
):
    '''This should be executed as many separate thread blocks with nthreads = 32
    As all 32 threads will run in a single warp, there is no need for synchronization
    steps, as the threads are all executed in lock step.
    "end" is actually one past the last element; compare ii < end'''

    shared_hts = cuda.shared.array((32, 12), numba.float64)
    shared_is_root = cuda.shared.array((32), numba.int32)

    pos = cuda.grid(1)
    warp_pos = pos & 31

    ii_ind = start + pos
    if ii_ind < end:
        for jj in range(12):
            # TO DO: minimize bank conflicts -- align memory reads
            shared_hts[warp_pos, jj] = hts[ii_ind, jj]
        shared_is_root[warp_pos] = is_root[ii_ind]
        myht = ht_load(shared_hts, warp_pos)
        parent = parent_ind[ii_ind].item()
        htchanged = False
        if parent != -1:
            parent_ht = ht_load(hts, parent)
            myht = ht_multiply(parent_ht, myht)
            htchanged = True
        myroot = shared_is_root[warp_pos]
        if warp_pos == 0 and cuda.blockIdx.x > 0 and not myroot:
            carry_ht = ht_load(hts_int, cuda.blockIdx.x - 1)
            #print("carry_ht", carry_ht)
            myht = ht_multiply(carry_ht, myht)
            htchanged = True
        if htchanged:
            ht_save(shared_hts, warp_pos, myht)
    cuda.syncthreads()

    # begin segmented scan
    offset = 1
    for jj in range(6):  #log2(256) == 8
        if warp_pos >= offset and ii_ind < end:
            if not myroot:
                prev_ht = ht_load(shared_hts, warp_pos - offset)
                prev_root = shared_is_root[warp_pos - offset]
        cuda.syncthreads()
        if warp_pos >= offset and ii_ind < end:
            if not myroot:
                myht = ht_multiply(prev_ht, myht)
                myroot |= prev_root
                ht_save(shared_hts, warp_pos, myht)
                shared_is_root[warp_pos] = myroot
        offset *= 2
        cuda.syncthreads()

    #save the HTs to global memory
    if ii_ind < end:
        for jj in range(12):
            hts[ii_ind, jj] = shared_hts[warp_pos, jj]


def get_devicendarray(t):
    import ctypes
    '''Convert a device-allocated pytorch tensor into a numba DeviceNDArray'''
    #print("get_devicendarray",t.type())
    if t.type() == 'torch.cuda.FloatTensor':
        ctx = cuda.cudadrv.driver.driver.get_context()
        mp = cuda.cudadrv.driver.MemoryPointer(
            ctx, ctypes.c_ulong(t.data_ptr()),
            t.numel() * 4
        )
        return cuda.cudadrv.devicearray.DeviceNDArray(
            t.size(), [i * 4 for i in t.stride()],
            numpy.dtype('float32'),
            gpu_data=mp,
            stream=torch.cuda.current_stream().cuda_stream
        )
    elif t.type() == 'torch.cuda.DoubleTensor':
        ctx = cuda.cudadrv.driver.driver.get_context()
        mp = cuda.cudadrv.driver.MemoryPointer(
            ctx, ctypes.c_ulong(t.data_ptr()),
            t.numel() * 8
        )
        return cuda.cudadrv.devicearray.DeviceNDArray(
            t.size(), [i * 8 for i in t.stride()],
            numpy.dtype('float64'),
            gpu_data=mp,
            stream=torch.cuda.current_stream().cuda_stream
        )
    else:
        # We're using the numba cuda simulator; this will let us modify the underlying
        # numpy array in numba on the CPU. Neat!
        return t.numpy()


def segscan_hts_gpu(hts_ko, refold_data):
    rd = refold_data
    stream = cuda.stream()

    hts_ro_d = cuda.device_array((rd.natoms, 12), dtype=numpy.float64)

    nblocks = (rd.natoms - 1) // 512 + 1
    reorder_starting_hts[nblocks, 512, stream](
        rd.natoms, hts_ko, hts_ro_d, rd.ki2ri_d
    )

    # for each depth, run a separate segmented scan
    for iirange in rd.refold_atom_range_for_depth:
        #print(iirange)
        segscan_ht_interval_one_thread_block[1, 256, stream](
            hts_ro_d, rd.is_root_ro_d, rd.non_subpath_parent_ro_d, rd.natoms,
            iirange[0], iirange[1]
        )

    reorder_final_hts[nblocks, 512, stream](
        rd.natoms, hts_ko, hts_ro_d, rd.ki2ri_d
    )


def segscan_hts_gpu2(hts_ko, refold_data):
    rd = refold_data
    stream = cuda.stream()

    hts_ro_d = cuda.device_array((rd.natoms, 12), dtype=numpy.float64)
    nblocks32 = (rd.natoms - 1) // 32 + 1
    hts_inter_d = cuda.device_array((nblocks32, 12), dtype=numpy.float64)
    is_root_inter_d = cuda.device_array((nblocks32), dtype=numpy.bool)

    nblocks512 = (rd.natoms - 1) // 512 + 1

    reorder_starting_hts[nblocks512, 512, stream](
        rd.natoms, hts_ko, hts_ro_d, rd.ki2ri_d
    )

    # for each depth, run a separate segmented scan
    for iirange in rd.refold_atom_range_for_depth:
        # Scan proceeds in three stages: a first stage where each thread block
        # is performed by a single warp w/ no need for thread synchronization
        # (since the threads are guaranteed lock step), and the result of each
        # thread block is written to an intermediate global array of HTs
        # as well as of is_root booleans (i.e. the segment start booleans
        # needed for segmented scan). The second step is performed in a single
        # thread block that performs segmented scan on each of the
        # intermediates. These are then written back to the intermediate HT
        # array in global memory. The third and final step is to run
        # another segmented scan very similar to the first one, but this time
        # with thread 0 of the block folding in the HT from the global array
        # of intermediate HTs computed in step two.

        #print(iirange)
        nblocks_step1 = (iirange[1] - iirange[0] - 1) // 32 + 1
        segscan_ht_interval_many_thread_blocks_1[nblocks_step1, 32, stream](
            hts_ro_d, hts_inter_d, rd.is_root_ro_d, is_root_inter_d,
            rd.non_subpath_parent_ro_d, rd.natoms, iirange[0], iirange[1]
        )

        #print("intermediate hts after stage 1", hts_inter_d.copy_to_host() )

        nblocks_step2 = (iirange[1] - iirange[0] - 1) // 256 + 1
        segscan_ht_interval_one_thread_block_2[nblocks_step2, 256, stream](
            hts_inter_d, is_root_inter_d, nblocks_step1
        )
        #print("intermediate hts after stage 2", hts_inter_d.copy_to_host() )

        segscan_ht_interval_many_thread_blocks_3[nblocks_step1, 32, stream](
            hts_ro_d, hts_inter_d, rd.is_root_ro_d, rd.non_subpath_parent_ro_d,
            rd.natoms, iirange[0], iirange[1]
        )

    reorder_final_hts[nblocks512, 512, stream](
        rd.natoms, hts_ko, hts_ro_d, rd.ki2ri_d
    )


@numba.jit(nopython=True)
def mark_path_children_and_count_nonpath_children(
        natoms, parent_ko, subpath_child_ko, n_nonpath_children_ko,
        is_subpath_root_ko, is_subpath_leaf_ko
):
    for ii in range(natoms - 1, -1, -1):
        ii_parent = parent_ko[ii]
        if ii == ii_parent:
            is_subpath_root_ko[ii] = True
        elif subpath_child_ko[ii_parent] != ii:
            n_nonpath_children_ko[ii_parent] += 1
            is_subpath_root_ko[ii] = True
        is_subpath_leaf_ko[ii] = subpath_child_ko[ii] == -1
    #print("rd.is_subpath_root_ko")
    #print(rd.is_subpath_root_ko)


@numba.jit(nopython=True)
def list_nonpath_children(
        natoms, is_subpath_root_ko, parent_ko, non_path_children_ko
):
    count_n_nonfirst_children = numpy.zeros((natoms), dtype=numpy.int32)
    for ii in range(natoms):
        if is_subpath_root_ko[ii]:
            ii_parent = parent_ko[ii]
            if ii_parent == ii:
                continue
            ii_child_ind = count_n_nonfirst_children[ii_parent]
            non_path_children_ko[ii_parent, ii_child_ind] = ii
            count_n_nonfirst_children[ii_parent] += 1
    #print("rd.non_path_children_ko"); print(non_path_children_ko)


@numba.jit(nopython=True)
def find_derivsum_path_depths(
        natoms, subpath_child_ko, derivsum_path_depth_ko, non_path_children_ko,
        is_subpath_root_ko, subpath_length_ko
):
    for ii in range(natoms - 1, -1, -1):
        # my depth is the larger of my first child's depth, or
        # my other children's laregest depth + 1
        ii_depth = 0
        ii_child = subpath_child_ko[ii]
        if ii_child != -1:
            ii_depth = derivsum_path_depth_ko[ii_child]
            #print(ii,"child",ii_child,"depth",ii_depth)
        for other_child in non_path_children_ko[ii, :]:
            if other_child == -1:
                continue
            other_child_depth = derivsum_path_depth_ko[other_child]
            #print("ii",ii,"other_child",other_child,"other_child_depth",other_child_depth)
            if ii_depth < other_child_depth + 1:
                ii_depth = other_child_depth + 1
        #print(ii,"depth",ii_depth)
        derivsum_path_depth_ko[ii] = ii_depth

        # if this is the root of a derivsum path (remember, paths are summed
        # leaf to root), then visit all of the nodes on the path and mark them
        # with my depth. I'm not sure this is necessary
        if is_subpath_root_ko[ii]:
            next_node = subpath_child_ko[ii]
            path_length = 1
            leaf_node = ii
            while next_node != -1:
                leaf_node = next_node
                derivsum_path_depth_ko[next_node] = ii_depth
                next_node = subpath_child_ko[next_node]
                if next_node != -1:
                    leaf_node = next_node
                path_length += 1
            subpath_length_ko[ii] = path_length
            subpath_length_ko[leaf_node] = path_length


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


def send_derivsum_data_to_gpu(
        natoms, ki2dsi, is_leaf_dso, non_path_children_dso
):
    ki2dsi_d = cuda.to_device(ki2dsi)
    is_leaf_dso_d = cuda.to_device(is_leaf_dso)
    non_path_children_dso_d = \
        cuda.to_device(non_path_children_dso)
    return ki2dsi_d, is_leaf_dso_d, non_path_children_dso_d


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


# f1f2 summation should probably be at double precision
#@cuda.jit('float64[:,:], int32[:,:], boolean[:], int32, int32, int32')
@cuda.jit
def segscan_f1f2s_up_tree(
        f1f2s_dso, prior_children, is_leaf, start, end, n_derivsum_nodes
):
    shared_f1f2s = cuda.shared.array((512, 6), numba.float64)
    shared_is_leaf = cuda.shared.array((512), numba.int32)

    pos = cuda.grid(1)
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


def segscan_f1f2s_gpu(f1f2s_ko, refold_data):
    rd = refold_data
    f1f2s_dso_d = cuda.device_array((rd.natoms, 6), dtype="float64")

    nblocks = (rd.natoms - 1) // 512 + 1
    reorder_starting_f1f2s[nblocks, 512](
        rd.natoms, f1f2s_ko, f1f2s_dso_d, rd.ki2dsi_d
    )

    for iirange in rd.derivsum_atom_range_for_depth:
        segscan_f1f2s_up_tree[1, 512](
            f1f2s_dso_d, rd.non_path_children_dso_d, rd.is_leaf_dso_d,
            iirange[0], iirange[1], rd.natoms
        )

    reorder_final_f1f2s[nblocks, 512](
        rd.natoms, f1f2s_ko, f1f2s_dso_d, rd.ki2dsi_d
    )
