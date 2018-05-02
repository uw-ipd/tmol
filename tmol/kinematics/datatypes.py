import enum
import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

from tmol.types.attrs import ValidateAttrs, ConvertAttrs


# types of kintree nodes
class NodeType(enum.IntEnum):
    root = 0
    jump = enum.auto()
    bond = enum.auto()


# convenience indexing methods
class BondDOFs(enum.IntEnum):
    phi_p = 0
    theta = enum.auto()
    d = enum.auto()
    phi_c = enum.auto()


class JumpDOFs(enum.IntEnum):
    RBx = 0
    RBy = enum.auto()
    RBz = enum.auto()
    RBdel_alpha = enum.auto()
    RBdel_beta = enum.auto()
    RBdel_gamma = enum.auto()
    RBalpha = enum.auto()
    RBbeta = enum.auto()
    RBgamma = enum.auto()


# data structure describing atom level kinematics of a molecular system
@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinTree(TensorGroup, ConvertAttrs):
    #fd: alex, these are all 1xN tensors.
    #    this requires .squeeze() operations throughout the code
    #    is there a better way to do this?
    id: Tensor(torch.int)[:, 1]
    doftype: Tensor(torch.int)[:, 1]
    parent: Tensor(torch.long)[:, 1]  # used as an index so long
    frame: Tensor(torch.long)[:, 3]  # used as an index so long

    # set a slice from 6 elts
    def __setitem__(self, idx, value):
        self.id[idx] = value[0]
        self.doftype[idx] = value[1]
        self.parent[idx] = value[2]
        self.frame[idx][:] = torch.tensor(value[3:])

    def __len__(self):
        return self.id.shape[0]

    @classmethod
    def full(cls, nelts, fill_value):
        return cls(
            id=torch.full((nelts, 1), fill_value),
            doftype=torch.full((nelts, 1), fill_value),
            parent=torch.full((nelts, 1), fill_value),
            frame=torch.full((nelts, 3), fill_value),
        )


# data structure describing internal coordinates
@attr.s(auto_attribs=True, slots=True, frozen=True)
class DofView(TensorGroup, ConvertAttrs):
    dofs: Tensor(torch.double)[:, 9]

    def __setitem__(self, idx, value):
        self.dofs[idx] = value.dofs[idx]

    def __len__(self):
        return self.dofs.shape[0]

    def bondDofView(self):
        return self.dofs[:, :4]

    def jumpDofView(self):
        return self.dofs

    def rawDofView(self):
        return self.dofs

    def clone(self):
        return DofView(dofs=self.dofs.clone())

    @classmethod
    def full(cls, nelts, fill_value):
        return cls(dofs=torch.full((nelts, 9), fill_value), )
