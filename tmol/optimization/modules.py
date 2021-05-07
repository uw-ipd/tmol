import torch
import math

from tmol.kinematics.metadata import DOFTypes
from tmol.score.coordinates import KinematicAtomicCoordinateProvider
from tmol.system.kinematics import KinematicDescription
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
        self.dofs = torch.nn.Parameter(coords)

    def forward(self):
        return self.score_system.intra_total(self.dofs)


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


# torsion space minimization
class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, ubq_system, torch_device):
        super(TorsionalEnergyNetwork, self).__init__()

        # Initialize kinematic tree for the system
        sys_kin = KinematicDescription.for_system(
            ubq_system.bonds, ubq_system.torsion_metadata
        )
        kintree = sys_kin.kintree.to(torch_device)
        dofmetadata = sys_kin.dof_metadata.to(torch_device)
        # compute dofs from xyzs
        dofs = sys_kin.extract_kincoords(ubq_system.coords).to(torch_device)
        system_size = ubq_system.system_size

        self.score_system = score_system
        self.kintree = kintree
        self.system_size = system_size

        # todo: make this a configurable parameter
        #   (for now it defaults to torsion minimization)
        dofmetadata = dofmetadata
        dofmask = dofmetadata[dofmetadata.dof_type == DOFTypes.bond_torsion]
        self.mask = (dofmask.node_idx, dofmask.dof_idx)

        # parameters
        self.dofs = torch.nn.Parameter(dofs)

        # self.dofs = DOFMaskingFunc.apply(self.dofs, self.mask, dofs)

    def coords(self) -> Tensor[torch.float][:, :, 3]:
        """System cartesian atomic coordinates."""

        coords = torch.full(
            (self.system_size, 3),
            math.nan,
            dtype=self.dofs.dtype,
            layout=self.dofs.layout,
            device=self.dofs.device,
            requires_grad=False,
        )

        idIdx = self.kintree.id[1:].to(dtype=torch.long)
        coords[idIdx] = self.dofs[1:]

        return coords.to(torch.float)[None, ...]

    def forward(self):
        return self.score_system.intra_total(self.coords())
