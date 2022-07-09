import torch
import attr

from tmol.system.kinematics import KinematicDescription
from tmol.system.score_support import kincoords_to_coords

# modules for cartesian and torsion-space optimization
#
# unlike typical NN training, the model is fixed and we want to optimize inputs
#   therefore, the coordinates are parameters
#   there are no model inputs
#
# potentially a dof mask could be added here (?)
#  - or we might want to keep that with dof creation


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


# cartesian space minimization
class CartesianEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, coords, coord_mask=None):
        super(CartesianEnergyNetwork, self).__init__()

        self.score_system = score_system
        self.coord_mask = coord_mask

        self.full_coords = coords
        if self.coord_mask is None:
            self.masked_coords = torch.nn.Parameter(coords)
        else:
            self.masked_coords = torch.nn.Parameter(coords[self.coord_mask])

    def forward(self):
        self.full_coords = DOFMaskingFunc.apply(
            self.masked_coords, self.coord_mask, self.full_coords
        )
        return self.score_system.intra_total(self.full_coords)


def torsional_energy_network_from_system(
    score_system, residue_system, dof_mask=None, device=None
):
    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(
        residue_system.bonds, residue_system.torsion_metadata
    )
    if not device:
        device = torch.device("cpu")
    kintree = sys_kin.kintree.to(device)

    # compute dofs from xyzs
    dofs = sys_kin.extract_kincoords(residue_system.coords).to(device)
    system_size = residue_system.system_size

    return TorsionalEnergyNetwork(
        score_system, dofs, kintree, system_size, dof_mask=dof_mask
    )


# torsion space minimization
class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, dofs, kintree, system_size, dof_mask=None):
        super(TorsionalEnergyNetwork, self).__init__()

        self.score_system = score_system
        self.kintree = kintree

        # register buffers so they get moved to GPU with module
        for i, j in attr.asdict(kintree).items():
            self.register_buffer(i, j)
        self.register_buffer("dof_mask", dof_mask)
        self.register_buffer("full_dofs", dofs)
        self.system_size = system_size

        if self.dof_mask is None:
            self.masked_dofs = torch.nn.Parameter(dofs)
        else:
            self.masked_dofs = torch.nn.Parameter(dofs[self.dof_mask])

    def coords(self):
        self.full_dofs = DOFMaskingFunc.apply(
            self.masked_dofs, self.dof_mask, self.full_dofs
        )
        return kincoords_to_coords(self.full_dofs, self.kintree, self.system_size)

    def forward(self):
        return self.score_system.intra_total(self.coords())
