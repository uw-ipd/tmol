import enum
import torch
import attr
import typing
import numpy
import numba

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

    parent_ko: numpy.array = attr.ib(init=False)
    child_on_subpath_ko: numpy.array = attr.ib(init=False) #numpy.ones((self.natoms),dtype="int32")*-1
    len_longest_subpath_ko: Tensor(torch.long)[...] = attr.ib(init=False)# numpy.zeros((self.natoms),dtype="int32")
    subpath_root_ko: numpy.array = attr.ib(init=False) #numpy.full((self.natoms),True,dtype="boolean")
    atom_depth_ko: numpy.array = attr.ib(init=False)  #numpy.zeros((self.natoms),dtype="int32")
    depth_offsets: numpy.array = None

    def __attrs_post_init__(self):
        self.ri2ki = numpy.ones((self.natoms), dtype="int32")*-1
        self.ki2ri = numpy.ones((self.natoms), dtype="int32")*-1
        
        self.parent_ko = numpy.zeros((self.natoms),dtype="int32")
        self.child_on_subpath_ko = numpy.ones((self.natoms),dtype="int32")*-1
        self.len_longest_subpath_ko = numpy.zeros((self.natoms),dtype="int32")
        self.subpath_root_ko = numpy.full((self.natoms),True,dtype="bool")
        self.atom_depth_ko = numpy.zeros((self.natoms),dtype="int32")
        

def identify_longest_subpaths(refold_data):
    '''Visit all children before visiting the parent, identifying the length of the longest
    subpath rooted at that parent, and the child along that path'''
    __identify_longest_subpaths(
        refold_data.natoms, refold_data.parent_ko, refold_data.len_longest_subpath_ko,
        refold_data.child_on_subpath_ko, refold_data.subpath_root_ko )

@numba.jit(nopython=True)
def __identify_longest_subpaths(natoms, parent, len_longest_subpath_ko, child_on_subpath_ko, subpath_root_ko):
    for ii in range(natoms-1,-1,-1):
        len_longest_subpath_ko[ii] += 1
        ii_subpath = len_longest_subpath_ko[ii]
        ii_parent = parent[ii]
        if len_longest_subpath_ko[ii_parent] < ii_subpath and ii_parent != ii:
            len_longest_subpath_ko[ii_parent] = ii_subpath
            child_on_subpath_ko[ii_parent] = ii
        subpath_root_ko[child_on_subpath_ko[ii]] = False

def recursively_identify_longest_subpaths(kintree, refold_data, kin_atom_ind):
    if kin_atom_ind >= refold_data.natoms: return
    rd = refold_data
    kt = kintree

    recursively_identify_longest_subpaths(kintree, refold_data, kin_atom_ind+1)

    rd.len_longest_subpath_ko[kin_atom_ind] += 1
    my_subpath = rd.len_longest_subpath_ko[kin_atom_ind]
    parent = kt.parent[kin_atom_ind]
    if rd.len_longest_subpath_ko[parent] < my_subpath and parent != kin_atom_ind:
        rd.len_longest_subpath_ko[parent] = my_subpath
        rd.child_on_subpath_ko[parent] = kin_atom_ind
    rd.subpath_root_ko[rd.child_on_subpath_ko[kin_atom_ind]] = False

def recursively_identify_path_depths(kintree, refold_data, kin_atom_ind):
    if kin_atom_ind >= refold_data.natoms: return
    rd = refold_data
    kt = kintree

    parent = kt.parent[kin_atom_ind]
    depth = rd.atom_depth_ko[parent]
    if rd.subpath_root_ko[kin_atom_ind] and parent != kin_atom_ind:
        depth += 1
    rd.atom_depth_ko[kin_atom_ind] = depth
    recursively_identify_path_depths(kintree, refold_data, kin_atom_ind+1)

def recursively_assign_refold_indices(kintree, refold_data, kin_atom_ind, refold_index):
    refold_data.ri2ki[refold_index] = kin_atom_ind
    refold_data.ki2ri[kin_atom_ind] = refold_index
    child = refold_data.child_on_subpath_ko[kin_atom_ind]
    if child != -1:
        recursively_assign_refold_indices(
            kintree, refold_data,
            child, refold_index+1 )

def determine_refold_indices(kintree, refold_data):
    refold_data.parent_ko[:] = kintree.parent
    #recursively_identify_longest_subpaths(kintree, refold_data, 0)
    identify_longest_subpaths(refold_data)
    recursively_identify_path_depths(kintree, refold_data, 0)

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

    subpath_roots = numpy.nonzero(rd.subpath_root_ko)[0]
    root_depths = rd.atom_depth_ko[subpath_roots]
    for ii in range(rd.ndepths):
        ii_roots = subpath_roots[root_depths == ii]
        count = rd.depth_offsets[ii]
        for jj in ii_roots:
            recursively_assign_refold_indices(kintree, refold_data, jj, count )
            count += rd.len_longest_subpath_ko[jj]

    assert numpy.all( rd.ri2ki != -1 )
    assert numpy.all( rd.ki2ri != -1 )
    



    
