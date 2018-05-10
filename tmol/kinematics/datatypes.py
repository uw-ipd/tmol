import enum
import torch
import attr

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
