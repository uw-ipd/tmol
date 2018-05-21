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


class NodeType(enum.IntEnum):
    """KinTree node types."""
    root = 0
    jump = enum.auto()
    bond = enum.auto()


@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinTree(TensorGroup, ConvertAttrs):
    """Representation of atom-level kinematics."""

    # Indices:
    # id = index for kin-atom into the coords array 
    # parent = kin-atom index of parent for each kin-atom 
    # frame_x = kin-atom index of self
    # frame_y = kin-atom index of parent
    # frame_z = kin-atom index of grandparent
    # backward kinematics routines are in tmol/kinematics/operations.py

    id: Tensor(torch.long)[...]  # used as an index so long
    doftype: Tensor(torch.int)[...]
    parent: Tensor(torch.long)[...]  # used as an index so long
    frame_x: Tensor(torch.long)[...]
    frame_y: Tensor(torch.long)[...]
    frame_z: Tensor(torch.long)[...]

    @classmethod
    @convert_args
    def node(
            cls,
            id: int,
            doftype: NodeType,
            parent: int,
            frame_x: int,
            frame_y: int,
            frame_z: int,
    ):
        """Construct a single node from element values."""
        return cls(
            id=torch.Tensor([id]),
            doftype=torch.Tensor([doftype]),
            parent=torch.Tensor([parent]),
            frame_x=torch.Tensor([frame_x]),
            frame_y=torch.Tensor([frame_y]),
            frame_z=torch.Tensor([frame_z]),
        )

    @classmethod
    def root_node(cls):
        """The global/root kinematic node at KinTree[0]."""
        return cls.node(
            id=-1,
            doftype=NodeType.root,
            parent=0,
            frame_x=0,
            frame_y=0,
            frame_z=0,
        )


@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinDOF(TensorGroup, ConvertAttrs):
    """Internal coordinate data.

    The KinDOF data structure holds two logical views: the "raw" view a
    sparsely populated [n,9] tensor of DOF values and a set of named property
    accessors providing access to specific entries within this array. This is
    logically equivalent a C union datatype, the interpretation of an entry in
    the DOF buffer depends on the type of the corresponding KinTree entry.
    """

    raw: Tensor(torch.double)[..., 9]

    @property
    def bond(self):
        return BondDOF(raw=self.raw[..., :4])

    @property
    def jump(self):
        return JumpDOF(raw=self.raw[..., :9])

    def clone(self):
        return KinDOF(raw=self.raw.clone())


class BondDOFTypes(enum.IntEnum):
    """Indices of bond dof types within KinDOF.raw."""
    phi_p = 0
    theta = enum.auto()
    d = enum.auto()
    phi_c = enum.auto()


class JumpDOFTypes(enum.IntEnum):
    """Indices of jump dof types within KinDOF.raw."""
    RBx = 0
    RBy = enum.auto()
    RBz = enum.auto()
    RBdel_alpha = enum.auto()
    RBdel_beta = enum.auto()
    RBdel_gamma = enum.auto()
    RBalpha = enum.auto()
    RBbeta = enum.auto()
    RBgamma = enum.auto()


@attr.s(auto_attribs=True, slots=True, frozen=True)
class BondDOF(TensorGroup, ConvertAttrs):
    """A bond dof view of KinDOF."""
    raw: Tensor(torch.double)[..., 4]

    @property
    def phi_p(self):
        return self.raw[..., BondDOFTypes.phi_p]

    @property
    def theta(self):
        return self.raw[..., BondDOFTypes.theta]

    @property
    def d(self):
        return self.raw[..., BondDOFTypes.d]

    @property
    def phi_c(self):
        return self.raw[..., BondDOFTypes.phi_c]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class JumpDOF(TensorGroup, ConvertAttrs):
    """A jump dof view of KinDOF."""
    raw: Tensor(torch.double)[..., 9]

    @property
    def RBx(self):
        return self.raw[..., JumpDOFTypes.RBx]

    @property
    def RBy(self):
        return self.raw[..., JumpDOFTypes.RBy]

    @property
    def RBz(self):
        return self.raw[..., JumpDOFTypes.RBz]

    @property
    def RBdel_alpha(self):
        return self.raw[..., JumpDOFTypes.RBdel_alpha]

    @property
    def RBdel_beta(self):
        return self.raw[..., JumpDOFTypes.RBdel_beta]

    @property
    def RBdel_gamma(self):
        return self.raw[..., JumpDOFTypes.RBdel_gamma]

    @property
    def RBalpha(self):
        return self.raw[..., JumpDOFTypes.RBalpha]

    @property
    def RBbeta(self):
        return self.raw[..., JumpDOFTypes.RBbeta]

    @property
    def RBgamma(self):
        return self.raw[..., JumpDOFTypes.RBgamma]

@attr.s(auto_attribs=True)
class RefoldData:
    natoms: int
    ndepths: int = 0
    ri2ki: numpy.array = attr.ib(init=False)# numpy.ones((self.natoms), dtype="int32")*-1
    ki2ri: numpy.array = attr.ib(init=False)# numpy.ones((self.natoms), dtype="int32")*-1

    # Data used for forward kinematics
    parent_ko: numpy.array = attr.ib(init=False)
    parent_ro: numpy.array = attr.ib(init=False)
    child_on_subpath_ko: numpy.array = attr.ib(init=False) #numpy.ones((self.natoms),dtype="int32")*-1
    len_longest_subpath_ko: Tensor(torch.long)[...] = attr.ib(init=False)# numpy.zeros((self.natoms),dtype="int32")
    subpath_root_ko: numpy.array = attr.ib(init=False) #numpy.full((self.natoms),True,dtype="boolean")
    atom_depth_ko: numpy.array = attr.ib(init=False)  #numpy.zeros((self.natoms),dtype="int32")
    depth_offsets: numpy.array = None
    atom_range_for_depth: typing.List[typing.Tuple[int, int]] = None
    subpath_root_ro: numpy.array = attr.ib(init=False) 

    # Data used for f1f2 summation
    n_derivsum_depths: int = 0
    is_derivsum_root_ko: numpy.array = attr.ib(init=False) # starts all false
    is_derivsum_leaf_ko: numpy.array = attr.ib(init=False) # starts all false
    derivsum_path_length_ko: numpy.array = attr.ib(init=False) # starts at 0
    is_leaf_dso: numpy.array = attr.ib(init=False)
    is_root_dso: numpy.array = attr.ib(init=False)
    derivsum_first_child_ko: numpy.array = attr.ib(init=False) # starts -1
    n_nonpath_children_ko = attr.ib(init=False) # starts 0
    derivsum_path_depth_ko = attr.ib(init=False) # starts 0
    derivsum_atom_range_for_depth: Typing.List[Typing.Tuple[int,int]] = None
    ki2dsi = numpy.array = attr.ib(init=False)
    dsi2ki = numpy.array = attr.ib(init=False)
    non_path_children_ko = attr.ib(init=False) # starts -1
    non_path_children_dso = attr.ib(init=False)


    hts_ro_d: numba.types.Array = None
    parent_ro_d: numba.types.Array = None
    is_root_d: numba.types.Array = None
    ri2ki_d: numba.types.Array = None
    ki2ri_d: numba.types.Array = None

    def __attrs_post_init__(self):
        self.ri2ki = numpy.ones((self.natoms), dtype="int32")*-1
        self.ki2ri = numpy.ones((self.natoms), dtype="int32")*-1
        
        self.parent_ko = numpy.zeros((self.natoms),dtype="int32")
        self.parent_ro = numpy.zeros((self.natoms),dtype="int32")
        self.child_on_subpath_ko = numpy.ones((self.natoms),dtype="int32")*-1
        self.len_longest_subpath_ko = numpy.zeros((self.natoms),dtype="int32")
        self.subpath_root_ko = numpy.full((self.natoms),True,dtype="bool")
        self.atom_depth_ko = numpy.zeros((self.natoms),dtype="int32")
        self.subpath_root_ro = numpy.full((self.natoms),True,dtype="bool")
        

def identify_longest_subpaths(refold_data):
    '''Visit all children before visiting the parent, identifying the length of the longest
    subpath rooted at that parent, and the child along that path'''
    __identify_longest_subpaths(
        refold_data.natoms, refold_data.parent_ko, refold_data.len_longest_subpath_ko,
        refold_data.child_on_subpath_ko, refold_data.subpath_root_ko )

@numba.jit(nopython=True)
def __identify_longest_subpaths(natoms, parent_ko, len_longest_subpath_ko, child_on_subpath_ko, subpath_root_ko):
    for ii in range(natoms-1,-1,-1):
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
        refold_data.subpath_root_ko)


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
def finalize_refold_indices(roots, depth_offset, child_on_subpath_ko, ri2ki, ki2ri):
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
    identify_longest_subpaths(refold_data)
    identify_path_depths(refold_data)
    #recursively_identify_path_depths(kintree, refold_data, 0)

    # ok, sum the path lengths at each depth
    rd = refold_data
    rd.ndepths = max(rd.atom_depth_ko)+1
    rd.depth_offsets = numpy.zeros((rd.ndepths),dtype="int32")
    numpy.add.at(
        rd.depth_offsets,
        rd.atom_depth_ko[ rd.subpath_root_ko ],
        rd.len_longest_subpath_ko[ rd.subpath_root_ko ] )
    rd.depth_offsets[1:] = numpy.cumsum(rd.depth_offsets)[:-1]
    rd.depth_offsets[0] = 0
    rd.atom_range_for_depth = []
    for i in range(rd.ndepths-1):
        rd.atom_range_for_depth.append(
            (rd.depth_offsets[i], rd.depth_offsets[i+1])
        )
    rd.atom_range_for_depth.append((rd.depth_offsets[-1],rd.natoms))
    subpath_roots = numpy.nonzero(rd.subpath_root_ko)[0]
    root_depths = rd.atom_depth_ko[subpath_roots]
    for ii in range(rd.ndepths):
        ii_roots = subpath_roots[root_depths == ii]
        finalize_refold_indices(
            ii_roots, rd.depth_offsets[ii], 
            rd.child_on_subpath_ko,
            rd.ri2ki, rd.ki2ri )

    assert numpy.all( rd.ri2ki != -1 )
    assert numpy.all( rd.ki2ri != -1 )

    rd.subpath_root_ro[:] = rd.subpath_root_ko[rd.ri2ki]
    rd.parent_ro = numpy.full((rd.natoms),-1,dtype="int32")
    rd.parent_ro[rd.subpath_root_ro] = rd.ki2ri[rd.parent_ko[rd.ri2ki][rd.subpath_root_ro]]
    rd.parent_ro[0] = -1


def determine_derivsum_indices(kintree, refold_data):
    mark_derivsum_first_children(refold_data):
    max_n_nonpath_children = max(rd.n_nonpath_children_ko)
    rd.non_path_children_ko = numpy.ones((rd.natoms,max_n_nonpath_children),dtype="int32")*-1
    list_non_first_derivsum_children(refold_data)
    find_derivsum_path_depths(refold_data)

    leaf_path_depths = refold_data.derivsum_path_depth_ko[
        refold_data.is_derivsum_leaf_ko
    ]
    leaf_path_lengths = refold_data.derivsum_path_length_ko[
        refold_data.is_derivsum_leaf_ko
    ]
    rd.n_derivsum_depths = leaf_path_depths[0]
    depth_offsets = numpy.zeros((rd.n_derivsum_depths),dtype="int32")
    numpy.add.at(
        depth_offsets,
        leaf_path_depths,
        leaf_path_lengths
    )
    depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
    depth_offsets[0] = 0
    rd.derivsum_atom_range_for_depth = []
    for ii in range(rd.n_derivsum_depths-1):
        rd.derivsum_atom_range_for_depth.append(
            (depth_offsets[ii], depth_offsets[ii+1])
        )
    derivsum_leaves = numpy.nonzero(rd.is_derivsum_leaf_ko)
    for ii in range(rd.n_derivsum_dpeths):
        ii_leaves = derivsum_leaves[leaf_path_depths == ii]
        finalize_derivsum_indices(
            ii_leaves, depth_offsets[ii],
            rd.derivsum_first_child_ko,
            rd.ki2dsi, rd.dsi2ki )
    assert numpy.all( rd.ki2dsi != -1 )
    assert numpy.all( rd.dsi2ki != -1 )
    
    

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
def reorder_starting_hts( natoms, hts_ko, hts_ro, ki2ri ):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ro[ri,i] = hts_ko[pos,i//4,i%4]

@cuda.jit
def reorder_final_hts( natoms, hts_ko, hts_ro, ki2ri ):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ko[pos,i//4,i%4] = hts_ro[ri,i]


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
        hts[pos, i//4, i%4] = ht[i]

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
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], numpy.dtype('float32'),
        gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)

def segscan_hts_gpu(hts_ko,refold_data):
    rd = refold_data

    nblocks = (rd.natoms-1) // 512 + 1
    reorder_starting_hts[nblocks,512](
        rd.natoms, hts_ko, rd.hts_ro_d, rd.ki2ri_d)

    # for each depth, run a separate segmented scan
    for iirange in rd.atom_range_for_depth:
        #print(iirange)
        segscan_ht_interval[1, 512](
            rd.hts_ro_d, rd.is_root_d, rd.parent_ro_d, rd.natoms, iirange[0],
            iirange[1]
        )

    reorder_final_hts[nblocks,512](
        rd.natoms, hts_ko, rd.hts_ro_d, rd.ki2ri_d)


def mark_derivsum_first_children(refold_data):
    rd = refold_data
    for ii in range(rd,natoms-1,-1,-1):
        ii_parent = rd.parent_ko[ii]
        if rd.derivsum_first_child_ko[ii_parent] != -1:
            rd.derivsum_first_child_ko[ii_parent] = ii
        else:
            rd.n_nonpath_children_ko[ii_parent] += 1
            rd.is_derivsum_root_ko[ii] = True
        rd.is_derivsum_leaf_ko[ii] = rd.first_child_ko[ii] == -1

def list_non_first_derivsum_children(refold_data):
    rd = refold_data
    count_n_nonfirst_children = numpy.zeros((rd.natoms),dtype=numpy.int32)
    for ii in range(rd.natoms):
        if rd.is_derivsum_root_ko[ii]:
            ii_parent = rd.parent_ko[ii]
            if ii_parent == ii: continue
            ii_child_ind = count_n_nonfirst_children[ii_parent]
            rd.non_path_children_ko[ii_parent,ii_child_ind] = ii
            count_n_nonfirst_children[ii_parent] += 1

def find_derivsum_path_depths(refold_data):
    rd = refold_data
    for ii in range(rd.natoms-1,-1,-1):
        # my depth is the larger of my first child's depth, or
        # my other children's laregest depth + 1
        ii_depth = 0
        ii_child = rd.first_child_ko[ii]
        if ii_child == -1:
            ii_depth = rd.derivsum_depth_ko[ii_child]
        for other_child in rd.non_path_children_ko[ii]:
            if other_child == -1: continue
            other_child_depth = rd.derivsum_depth_ko[other_child]
            if ii_depth < other_child_depth + 1:
                ii_depth = other_child_depth + 1
        rd.derivsum_depth_ko[ii] = ii_depth

        # if this is the root of a derivsum path (remember, paths are summed
        # leaf to root), then visit all of the nodes on the path and mark them
        # with my depth. I'm not sure this is necessary
        if rd.is_derivsum_root_ko[ii]:
            next_node = rd.derivsum_first_child[ii]
            path_length = 1
            leaf_node
            while next_node != -1:
                rd.derivsum_depth_ko[next_node] = ii_depth
                next_node = rd.dervisum_first_child[next_node]
                if next_node != -1:
                    leaf_node = next_node
                path_length += 1
            rd.derivsum_path_length[ii] = path_length
            rd.derivsum_path_length[leaf_node] = path_length


@numba.jit(nopython=True)
def finalize_derivsum_indices( leaves, start_ind, first_child, ki2dsi, dsi2ki ):
    count = start_ind
    for leaf in leaves:
        nextatom = leaf
        while nextatom != -1:
            dsi2ki[count] = nextatom
            ki2dsi[nextatom] = count
            nextatom = first_child[nextatom]
            count += 1
            
        
