import torch
import math

from tmol.kinematics.metadata import DOFTypes
from tmol.system.kinematics import KinematicDescription
from tmol.system.score_support import kincoords_to_coords
from tmol.types.torch import Tensor

# modules for cartesian and torsion-space optimization
#
# unlike typical NN training, the model is fixed and we want to optimize inputs
#   therefore, the coordinates are parameters
#   there are no model inputs
#
# potentially a dof mask could be added here (?)
#  - or we might want to keep that with dof creation


# cartesian space minimization
class CartesianEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, coords):
        super(CartesianEnergyNetwork, self).__init__()

        # scoring graph
        self.score_system = score_system

        # parameters
        self.coords = torch.nn.Parameter(coords)

    def forward(self):
        return self.score_system.intra_total(self.coords)


# mask out relevant dofs to the minimizer
class DOFMaskingFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fg, mask, bg):
        ctx.mask = mask
        ctx.fg = fg
        bg[mask] = fg
        return bg

    @staticmethod
    def backward(ctx, grad_output):
        grad = torch.zeros_like(ctx.fg)
        grad = grad_output[ctx.mask]
        return grad, None, None


def torsional_energy_network_from_system(score_system, residue_system, torch_device):
    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(
        residue_system.bonds, residue_system.torsion_metadata
    )
    kintree = sys_kin.kintree.to(torch_device)

    # compute dofs from xyzs
    dofs = sys_kin.extract_kincoords(residue_system.coords).to(torch_device)
    system_size = residue_system.system_size

    return TorsionalEnergyNetwork(score_system, dofs, kintree, system_size)


# torsion space minimization
class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, dofs, kintree, system_size, mask=None):
        super(TorsionalEnergyNetwork, self).__init__()

        self.score_system = score_system
        self.kintree = kintree
        self.mask = mask
        self.system_size = system_size

        self.full_dofs = dofs
        if self.mask is None:
            self.masked_dofs = torch.nn.Parameter(dofs)
        else:
            self.masked_dofs = torch.nn.Parameter(dofs[self.mask])

    def coords(self):
        return kincoords_to_coords(self.masked_dofs, self.kintree, self.system_size)

    def forward(self):
        # self.masked_dofs = DOFMaskingFunc.apply(self.masked_dofs, self.mask, self.full_dofs)
        return self.score_system.intra_total(self.coords())
