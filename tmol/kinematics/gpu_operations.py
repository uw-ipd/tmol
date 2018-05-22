import enum
import torch
import attr
import typing
import numpy
import numba
from numba import cuda

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

from tmol.types.attrs import ConvertAttrs
from tmol.types.functional import convert_args



@attr.s(auto_attribs=True)
class RefoldData:
    natoms: int
    ndepths: int = 0
    ri2ki: numpy.array = attr.ib(
        init=False
    )  # numpy.ones((self.natoms), dtype="int32")*-1
    ki2ri: numpy.array = attr.ib(
        init=False
    )  # numpy.ones((self.natoms), dtype="int32")*-1

    # Data used for forward kinematics
    parent_ko: numpy.array = attr.ib(init=False)
    parent_ro: numpy.array = attr.ib(init=False)
    branching_factor_ko: numpy.array = attr.ib(init=False)
    branchiest_child_ko: numpy.array = attr.ib(init=False)
    child_on_subpath_ko: numpy.array = attr.ib(init=False)
    len_longest_subpath_ko: Tensor(torch.long)[...] = attr.ib(init=False)
    subpath_root_ko: numpy.array = attr.ib(init=False)
    atom_depth_ko: numpy.array = attr.ib(init=False)
    depth_offsets: numpy.array = None
    atom_range_for_depth: typing.List[typing.Tuple[int, int]] = None
    subpath_root_ro: numpy.array = attr.ib(init=False)

    # Data used for f1f2 summation
    n_derivsum_depths: int = 0
    is_derivsum_root_ko: numpy.array = attr.ib(init=False)  # starts all false
    is_derivsum_leaf_ko: numpy.array = attr.ib(init=False)  # starts all false
    derivsum_path_length_ko: numpy.array = attr.ib(init=False)  # starts at 0
    is_leaf_dso: numpy.array = attr.ib(init=False)
    derivsum_first_child_ko: numpy.array = attr.ib(init=False)  # starts -1
    n_nonpath_children_ko: numpy.array = attr.ib(init=False)  # starts 0
    derivsum_path_depth_ko: numpy.array = attr.ib(init=False)  # starts 0
    derivsum_atom_range_for_depth: typing.List[typing.Tuple[int, int]] = None
    ki2dsi: numpy.array = attr.ib(init=False)
    dsi2ki: numpy.array = attr.ib(init=False)
    non_path_children_ko: numpy.array = attr.ib(init=False)  # starts -1
    non_path_children_dso: numpy.array = attr.ib(init=False)

    hts_ro_d: numba.types.Array = None
    parent_ro_d: numba.types.Array = None
    is_root_d: numba.types.Array = None
    ri2ki_d: numba.types.Array = None
    ki2ri_d: numba.types.Array = None

    ki2dsi_d: numba.types.Array = None
    f1f2s_dso_d: numba.types.Array = None
    is_leaf_dso_d: numba.types.Array = None
    non_path_children_dso_d: numba.types.Array = None

    def __attrs_post_init__(self):
        self.ri2ki = numpy.full((self.natoms), -1, dtype="int32")
        self.ki2ri = numpy.full((self.natoms), -1, dtype="int32")

        self.parent_ko = numpy.zeros((self.natoms), dtype="int32")
        self.parent_ro = numpy.zeros((self.natoms), dtype="int32")
        self.branching_factor_ko = numpy.full((self.natoms), -1, dtype="int32")
        self.branchiest_child_ko = numpy.full((self.natoms), -1, dtype="int32")
        self.child_on_subpath_ko = numpy.ones((self.natoms), dtype="int32"
                                              ) * -1
        self.len_longest_subpath_ko = numpy.zeros((self.natoms), dtype="int32")
        self.subpath_root_ko = numpy.full((self.natoms), True, dtype="bool")
        self.atom_depth_ko = numpy.zeros((self.natoms), dtype="int32")
        self.subpath_root_ro = numpy.full((self.natoms), True, dtype="bool")

        self.is_derivsum_root_ko = numpy.full((self.natoms),
                                              False,
                                              dtype="bool")
        self.is_derivsum_leaf_ko = numpy.full((self.natoms),
                                              False,
                                              dtype="bool")
        self.derivsum_path_length_ko = numpy.full((self.natoms),
                                                  0,
                                                  dtype="int32")
        self.is_leaf_dso = numpy.full((self.natoms), False, dtype="bool")
        self.is_root_dso = numpy.full((self.natoms), False, dtype="bool")
        self.derivsum_first_child_ko = numpy.full((self.natoms),
                                                  -1,
                                                  dtype="int32")
        self.n_nonpath_children_ko = numpy.full((self.natoms),
                                                0,
                                                dtype="int32")
        self.derivsum_path_depth_ko = numpy.full((self.natoms),
                                                 -1,
                                                 dtype="int32")
        self.derivsum_atom_range_for_depth = []
        self.ki2dsi = numpy.full((self.natoms), -1, dtype="int32")
        self.dsi2ki = numpy.full((self.natoms), -1, dtype="int32")
        self.non_path_children_ko = numpy.full((self.natoms),
                                               -1,
                                               dtype="int32")
        self.non_path_children_dso = numpy.full((self.natoms),
                                                -1,
                                                dtype="int32")


def compute_branching_factor(refold_data):
    __compute_branching_factor(
        refold_data.natoms, refold_data.parent_ko,
        refold_data.branching_factor_ko, refold_data.branchiest_child_ko
    )
    #print("refold_data.branching_factor_ko", refold_data.branching_factor_ko)
    #print("refold_data.branchiest_child_ko", refold_data.branchiest_child_ko)


@numba.jit(nopython=True)
def __compute_branching_factor(
        natoms, parent, branching_factor, branchiest_child
):
    for ii in range(natoms - 1, -1, -1):
        ii_bf = branching_factor[ii]
        if ii_bf == -1:
            ii_bf = 0
            branching_factor[ii] = ii_bf
        ii_parent = parent[ii]
        if ii == ii_parent: continue
        parent_bf = branching_factor[ii_parent]
        if parent_bf == -1:
            branching_factor[ii_parent] = ii_bf
            branchiest_child[ii_parent] = ii
        elif ii_bf >= parent_bf:
            branching_factor[ii_parent] = max(ii_bf, parent_bf + 1)
            branchiest_child[ii_parent] = ii


def identify_longest_subpaths(refold_data):
    '''Visit all children before visiting the parent, identifying the length of the longest
    subpath rooted at that parent, and the child along that path'''
    __identify_longest_subpaths(
        refold_data.natoms, refold_data.parent_ko,
        refold_data.len_longest_subpath_ko, refold_data.child_on_subpath_ko,
        refold_data.subpath_root_ko
    )


@numba.jit(nopython=True)
def __identify_longest_subpaths(
        natoms, parent_ko, len_longest_subpath_ko, child_on_subpath_ko,
        subpath_root_ko
):
    for ii in range(natoms - 1, -1, -1):
        len_longest_subpath_ko[ii] += 1
        ii_subpath = len_longest_subpath_ko[ii]
        ii_parent = parent_ko[ii]
        if len_longest_subpath_ko[ii_parent] < ii_subpath and ii_parent != ii:
            len_longest_subpath_ko[ii_parent] = ii_subpath
            child_on_subpath_ko[ii_parent] = ii
        subpath_root_ko[child_on_subpath_ko[ii]] = False


def identify_path_depths(refold_data):
    '''Breadth-first traversal of the tree where the depth for a node is one greater
    than the depth of its parent if it is the root of a subpath'''
    __identify_path_depths(
        refold_data.natoms, refold_data.parent_ko, refold_data.atom_depth_ko,
        refold_data.subpath_root_ko
    )


@numba.jit(nopython=True)
def __identify_path_depths(natoms, parent_ko, atom_depth_ko, subpath_root_ko):
    for ii in range(natoms):
        ii_parent = parent_ko[ii]
        ii_depth = atom_depth_ko[ii_parent]
        if subpath_root_ko[ii] and ii_parent != ii:
            ii_depth += 1
        atom_depth_ko[ii] = ii_depth


#def recursively_identify_path_depths(kintree, refold_data, kin_atom_ind):
#    if kin_atom_ind >= refold_data.natoms: return
#    rd = refold_data
#    kt = kintree
#
#    parent = kt.parent[kin_atom_ind]
#    depth = rd.atom_depth_ko[parent]
#    if rd.subpath_root_ko[kin_atom_ind] and parent != kin_atom_ind:
#        depth += 1
#    rd.atom_depth_ko[kin_atom_ind] = depth
#    recursively_identify_path_depths(kintree, refold_data, kin_atom_ind+1)

#def recursively_assign_refold_indices(kintree, refold_data, kin_atom_ind, refold_index):
#    refold_data.ri2ki[refold_index] = kin_atom_ind
#    refold_data.ki2ri[kin_atom_ind] = refold_index
#    child = refold_data.child_on_subpath_ko[kin_atom_ind]
#    if child != -1:
#        recursively_assign_refold_indices(
#            kintree, refold_data,
#            child, refold_index+1 )


@numba.jit(nopython=True)
def finalize_refold_indices(
        roots, depth_offset, child_on_subpath_ko, ri2ki, ki2ri
):
    count = depth_offset
    for root in roots:
        nextatom = root
        while nextatom != -1:
            ri2ki[count] = nextatom
            ki2ri[nextatom] = count
            nextatom = child_on_subpath_ko[nextatom]
            count += 1


def determine_refold_indices(kintree, refold_data):
    refold_data.parent_ko[:] = kintree.parent
    compute_branching_factor(refold_data)
    identify_longest_subpaths(refold_data)
    identify_path_depths(refold_data)
    #recursively_identify_path_depths(kintree, refold_data, 0)

    # ok, sum the path lengths at each depth
    rd = refold_data
    rd.ndepths = max(rd.atom_depth_ko) + 1
    rd.depth_offsets = numpy.zeros((rd.ndepths), dtype="int32")
    numpy.add.at(
        rd.depth_offsets, rd.atom_depth_ko[rd.subpath_root_ko],
        rd.len_longest_subpath_ko[rd.subpath_root_ko]
    )
    rd.depth_offsets[1:] = numpy.cumsum(rd.depth_offsets)[:-1]
    rd.depth_offsets[0] = 0
    rd.atom_range_for_depth = []
    for i in range(rd.ndepths - 1):
        rd.atom_range_for_depth.append(
            (rd.depth_offsets[i], rd.depth_offsets[i + 1])
        )
    rd.atom_range_for_depth.append((rd.depth_offsets[-1], rd.natoms))
    subpath_roots = numpy.nonzero(rd.subpath_root_ko)[0]
    root_depths = rd.atom_depth_ko[subpath_roots]
    for ii in range(rd.ndepths):
        ii_roots = subpath_roots[root_depths == ii]
        finalize_refold_indices(
            ii_roots, rd.depth_offsets[ii], rd.child_on_subpath_ko, rd.ri2ki,
            rd.ki2ri
        )

    assert numpy.all(rd.ri2ki != -1)
    assert numpy.all(rd.ki2ri != -1)

    rd.subpath_root_ro[:] = rd.subpath_root_ko[rd.ri2ki]
    rd.parent_ro = numpy.full((rd.natoms), -1, dtype="int32")
    rd.parent_ro[rd.subpath_root_ro
                 ] = rd.ki2ri[rd.parent_ko[rd.ri2ki][rd.subpath_root_ro]]
    rd.parent_ro[0] = -1


def determine_derivsum_indices(kintree, refold_data):
    rd = refold_data

    mark_derivsum_first_children(refold_data)
    max_n_nonpath_children = max(rd.n_nonpath_children_ko)
    rd.non_path_children_ko = numpy.full(
        (rd.natoms, max_n_nonpath_children), -1, dtype="int32")
    list_non_first_derivsum_children(refold_data)
    find_derivsum_path_depths(refold_data)

    leaf_path_depths = refold_data.derivsum_path_depth_ko[
        refold_data.is_derivsum_leaf_ko
    ]
    leaf_path_lengths = refold_data.derivsum_path_length_ko[
        refold_data.is_derivsum_leaf_ko
    ]
    rd.n_derivsum_depths = refold_data.derivsum_path_depth_ko[0] + 1
    depth_offsets = numpy.zeros((rd.n_derivsum_depths), dtype="int32")
    numpy.add.at(depth_offsets, leaf_path_depths, leaf_path_lengths)
    depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
    depth_offsets[0] = 0
    rd.derivsum_atom_range_for_depth = []
    for ii in range(rd.n_derivsum_depths - 1):
        rd.derivsum_atom_range_for_depth.append(
            (depth_offsets[ii], depth_offsets[ii + 1])
        )
    rd.derivsum_atom_range_for_depth.append((depth_offsets[-1],rd.natoms))
    derivsum_leaves = numpy.nonzero(rd.is_derivsum_leaf_ko)[0]
    for ii in range(rd.n_derivsum_depths):
        ii_leaves = derivsum_leaves[leaf_path_depths == ii]
        finalize_derivsum_indices(
            ii_leaves, depth_offsets[ii], rd.parent_ko, rd.is_derivsum_root_ko,
            rd.ki2dsi, rd.dsi2ki
        )

    #print("ki2dsi"); print(rd.ki2dsi)
    #print("dsi2ki"); print(rd.dsi2ki)

    assert numpy.all(rd.ki2dsi != -1)
    assert numpy.all(rd.dsi2ki != -1)

    rd.non_path_children_dso = numpy.full(
        rd.non_path_children_ko.shape, -1, dtype="int32"
    )
    for ii in range(rd.non_path_children_ko.shape[1]):
        child_exists = rd.non_path_children_ko[:, ii] != -1
        rd.non_path_children_dso[child_exists, ii] = rd.ki2dsi[
            rd.non_path_children_ko[child_exists, ii]
        ]
    # now all the identies of the children have been remapped, but they
    # are still in kintree order; so reorder them to derivsum order.
    rd.non_path_children_dso = rd.non_path_children_dso[rd.dsi2ki]
    rd.is_leaf_dso[:] = rd.is_derivsum_leaf_ko[rd.dsi2ki]


def send_refold_data_to_gpu(refold_data):
    rd = refold_data
    rd.hts_ro_d = cuda.to_device(
        numpy.zeros((rd.natoms, 12), dtype=numpy.float32)
    )
    rd.is_root_d = cuda.to_device(rd.subpath_root_ro)
    rd.ri2ki_d = cuda.to_device(rd.ri2ki)
    rd.ki2ri_d = cuda.to_device(rd.ki2ri)
    rd.parent_ro_d = cuda.to_device(rd.parent_ro)


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


@cuda.jit('float32[:,:], boolean[:], int32[:], int32, int32, int32')
def segscan_ht_interval(hts, is_root, parent_ind, natoms, start, end):
    # this should be executed as a single thread block with nthreads = 512
    # "end" is actually one past the last element; compare ii < end
    shared_hts = cuda.shared.array((512, 12), numba.float32)
    shared_is_root = cuda.shared.array((512), numba.int32)

    pos = cuda.grid(1)
    niters = (end - start - 1) // 512 + 1
    carry_ht = identity_ht()
    carry_is_root = False
    for ii in range(niters):
        ii_ind = ii * 512 + start + pos
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
        for jj in range(9):  #log2(512) == 9
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
            carry_ht = ht_load(shared_hts, 511)
            carry_is_root = shared_is_root[511]

        cuda.syncthreads()


def get_devicendarray(t):
    '''Convert a device-allocated pytorch tensor into a numba DeviceNDArray'''
    #print(t.type())
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
    else:
        # We're using the numba cuda simulator; this will let us modify the underlying
        # numpy array in numba on the CPU. Neat!
        return cuda.to_device(t.numpy())


def segscan_hts_gpu(hts_ko, refold_data):
    rd = refold_data

    nblocks = (rd.natoms - 1) // 512 + 1
    reorder_starting_hts[nblocks, 512](
        rd.natoms, hts_ko, rd.hts_ro_d, rd.ki2ri_d
    )

    # for each depth, run a separate segmented scan
    for iirange in rd.atom_range_for_depth:
        #print(iirange)
        segscan_ht_interval[1, 512](
            rd.hts_ro_d, rd.is_root_d, rd.parent_ro_d, rd.natoms, iirange[0],
            iirange[1]
        )

    reorder_final_hts[nblocks, 512](rd.natoms, hts_ko, rd.hts_ro_d, rd.ki2ri_d)


def mark_derivsum_first_children(refold_data):
    rd = refold_data
    rd.derivsum_first_child_ko[:] = rd.branchiest_child_ko
    for ii in range(rd.natoms - 1, -1, -1):
        ii_parent = rd.parent_ko[ii]
        if ii == ii_parent:
            rd.is_derivsum_root_ko[ii] = True
        elif rd.derivsum_first_child_ko[ii_parent] != ii:
            rd.n_nonpath_children_ko[ii_parent] += 1
            rd.is_derivsum_root_ko[ii] = True
        rd.is_derivsum_leaf_ko[ii] = rd.derivsum_first_child_ko[ii] == -1
    #print("rd.is_derivsum_root_ko")
    #print(rd.is_derivsum_root_ko)


def list_non_first_derivsum_children(refold_data):
    rd = refold_data
    count_n_nonfirst_children = numpy.zeros((rd.natoms), dtype=numpy.int32)
    for ii in range(rd.natoms):
        if rd.is_derivsum_root_ko[ii]:
            ii_parent = rd.parent_ko[ii]
            if ii_parent == ii: continue
            ii_child_ind = count_n_nonfirst_children[ii_parent]
            rd.non_path_children_ko[ii_parent, ii_child_ind] = ii
            count_n_nonfirst_children[ii_parent] += 1


def find_derivsum_path_depths(refold_data):
    rd = refold_data
    for ii in range(rd.natoms - 1, -1, -1):
        # my depth is the larger of my first child's depth, or
        # my other children's laregest depth + 1
        ii_depth = 0
        ii_child = rd.derivsum_first_child_ko[ii]
        if ii_child != -1:
            ii_depth = rd.derivsum_path_depth_ko[ii_child]
            #print(ii,"child",ii_child,"depth",ii_depth)
        for other_child in rd.non_path_children_ko[ii, :]:
            if other_child == -1: continue
            other_child_depth = rd.derivsum_path_depth_ko[other_child]
            #print("ii",ii,"other_child",other_child,"other_child_depth",other_child_depth)
            if ii_depth < other_child_depth + 1:
                ii_depth = other_child_depth + 1
        #print(ii,"depth",ii_depth)
        rd.derivsum_path_depth_ko[ii] = ii_depth

        # if this is the root of a derivsum path (remember, paths are summed
        # leaf to root), then visit all of the nodes on the path and mark them
        # with my depth. I'm not sure this is necessary
        if rd.is_derivsum_root_ko[ii]:
            next_node = rd.derivsum_first_child_ko[ii]
            path_length = 1
            leaf_node = ii
            while next_node != -1:
                leaf_node = next_node
                rd.derivsum_path_depth_ko[next_node] = ii_depth
                next_node = rd.derivsum_first_child_ko[next_node]
                if next_node != -1:
                    leaf_node = next_node
                path_length += 1
            rd.derivsum_path_length_ko[ii] = path_length
            rd.derivsum_path_length_ko[leaf_node] = path_length


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

def send_derivsum_data_to_gpu(refold_data):
    refold_data.ki2dsi_d = cuda.to_device(refold_data.ki2dsi)
    refold_data.f1f2s_dso_d = \
        cuda.to_device(numpy.zeros((refold_data.natoms,6),dtype="float64"))
    refold_data.is_leaf_dso_d = cuda.to_device(refold_data.is_leaf_dso)
    refold_data.non_path_children_dso_d = \
        cuda.to_device(refold_data.non_path_children_dso)

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
@cuda.jit('float64[:,:], int32[:,:], boolean[:], int32, int32, int32')
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
    nblocks = (rd.natoms - 1) // 512 + 1
    reorder_starting_f1f2s[nblocks, 512](
        rd.natoms, f1f2s_ko, rd.f1f2s_dso_d, rd.ki2dsi_d
    )

    for iirange in rd.derivsum_atom_range_for_depth:
        segscan_f1f2s_up_tree[1,512](
            rd.f1f2s_dso_d, rd.non_path_children_dso_d, rd.is_leaf_dso_d,
            iirange[0], iirange[1], rd.natoms
        )

    reorder_final_f1f2s[nblocks, 512](
        rd.natoms, f1f2s_ko, rd.f1f2s_dso_d, rd.ki2dsi_d
    )
