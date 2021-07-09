import torch

from .packed import PackedResidueSystem
from ..kinematics.dof_modules import CartesianDOFs, KinematicDOFs, KinematicOperation
from ..kinematics.operations import inverseKin

from .kinematics import KinematicDescription


@CartesianDOFs.get_from.register(PackedResidueSystem)
def cartesian_dofs_get_from(self: CartesianDOFs, system: PackedResidueSystem):
    self.coords = torch.nn.Parameter(
        torch.tensor(system.coords.reshape(1, len(system.coords), 3), dtype=torch.float)
    )

    return self


@CartesianDOFs.set_on.register(PackedResidueSystem)
def cartesian_dofs_set_on(self: CartesianDOFs, system: PackedResidueSystem):

    coords = self()
    assert coords.shape == (1, len(system.coords), 3)
    system.coords[:] = coords[0].numpy()

    return system


@KinematicOperation.build_for.register(PackedResidueSystem)
def kinematic_operation_build_for(system: PackedResidueSystem) -> KinematicOperation:
    sys_kin = KinematicDescription.for_system(system.system_size, system.bonds, (system.torsion_metadata,))
    kintree = sys_kin.kintree
    dof_metadata = sys_kin.dof_metadata

    return KinematicOperation(
        system_size=system.system_size, kintree=kintree, dof_metadata=dof_metadata
    )


@KinematicDOFs.get_from.register(PackedResidueSystem)
def kinematic_dofs_get_from(
    self: KinematicDOFs, system: PackedResidueSystem
) -> KinematicDOFs:
    kincoords = KinematicDescription(
        kintree=self.kinop.kintree, dof_metadata=self.kinop.dof_metadata
    ).extract_kincoords(system.coords)

    bkin = inverseKin(self.kinop.kintree, kincoords)

    self.full_dofs = bkin.raw.clone()
    self.dofs = torch.nn.Parameter(self.full_dofs[tuple(self.dof_mask)])

    return self


@KinematicDOFs.set_on.register(PackedResidueSystem)
def kinematic_dofs_set_on(
    self: KinematicDOFs, system: PackedResidueSystem
) -> PackedResidueSystem:

    coords = self()
    assert coords.shape == (1, len(system.coords), 3)
    system.coords[:] = coords[0].numpy()

    return system
