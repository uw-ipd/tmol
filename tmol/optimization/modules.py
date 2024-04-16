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


# cartesian space minimization
class CartesianEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, coords, coord_mask=None):
        super(CartesianEnergyNetwork, self).__init__()

        self.score_system = score_system

        self.full_coords = coords
        if coord_mask is None:
            coord_mask = torch.full(
                coords.shape[:-1], True, device=coords.device, dtype=torch.bool
            )
        self.masked_coords = torch.nn.Parameter(coords[coord_mask])
        self.mask = coord_mask

    def forward(self):
        self.full_coords = self.full_coords.detach()
        self.full_coords[self.mask] = self.masked_coords
        return self.score_system.intra_total(self.full_coords)


def torsional_energy_network_from_system(
    score_system, residue_system, dof_mask=None, device=None
):
    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(
        system_size=residue_system.system_size,
        bonds=residue_system.bonds,
        torsion_metadata=(residue_system.torsion_metadata,),
    )

    if not device:
        device = torch.device("cpu")
    kinforest = sys_kin.kinforest.to(device)

    # compute dofs from xyzs
    dofs = sys_kin.extract_kincoords(residue_system.coords).to(device)
    system_size = residue_system.system_size

    return TorsionalEnergyNetwork(
        score_system, dofs, kinforest, system_size, dof_mask=dof_mask
    )


# torsion space minimization
class TorsionalEnergyNetwork(torch.nn.Module):
    def __init__(self, score_system, dofs, kinforest, system_size, dof_mask=None):
        super(TorsionalEnergyNetwork, self).__init__()

        self.score_system = score_system
        self.kinforest = kinforest

        if dof_mask is None:
            dof_mask = torch.full(
                dofs.shape[:-1], True, device=dofs.device, dtype=torch.bool
            )

        self.system_size = system_size

        # register buffers so they get moved to GPU with module
        for i, j in attr.asdict(kinforest).items():
            self.register_buffer(i, j)
        self.register_buffer("dof_mask", dof_mask)
        self.register_buffer("full_dofs", dofs)

        self.masked_dofs = torch.nn.Parameter(dofs[self.dof_mask])

    def coords(self):
        self.full_dofs = self.full_dofs.detach()
        self.full_dofs[self.dof_mask] = self.masked_dofs
        return kincoords_to_coords(self.full_dofs, self.kinforest, self.system_size)

    def forward(self):
        return self.score_system.intra_total(self.coords())
