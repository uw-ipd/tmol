import numpy
import torch

from ..types.functional import validate_args

from ..kinematics.torch_op import KinematicOp
from ..kinematics.metadata import DOFTypes

from ..score.stacked_system import StackedSystem
from ..score.bonded_atom import BondedAtomScoreGraph

from ..score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from .packed import PackedResidueSystem
from .kinematics import KinematicDescription


@StackedSystem.factory_for.register(PackedResidueSystem)
@validate_args
def stack_params_for_system(system: PackedResidueSystem, **_):
    return dict(stack_depth=1, system_size=int(system.system_size))


@BondedAtomScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def bonded_atoms_for_system(
    system: PackedResidueSystem, drop_missing_atoms: bool = False, **_
):
    bonds = numpy.empty((len(system.bonds), 3), dtype=int)
    bonds[:, 0] = 0
    bonds[:, 1:] = system.bonds

    atom_types = system.atom_metadata["atom_type"].copy()[None, :]
    atom_elements = system.atom_metadata["atom_element"].copy()[None, :]
    atom_names = system.atom_metadata["atom_name"].copy()[None, :]
    res_indices = system.atom_metadata["residue_index"].copy()[None, :]
    res_names = system.atom_metadata["residue_name"].copy()[None, :]

    if drop_missing_atoms:
        atom_types[0, numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        bonds=bonds,
        atom_types=atom_types,
        atom_elements=atom_elements,
        atom_names=atom_names,
        res_indices=res_indices,
        res_names=res_names,
    )


@CartesianAtomicCoordinateProvider.factory_for.register(PackedResidueSystem)
@validate_args
def coords_for_system(
    system: PackedResidueSystem, device: torch.device, requires_grad: bool = True, **_
):
    """Extract constructor kwargs to initialize a `CartesianAtomicCoordinateProvider`"""

    stack_depth = 1
    system_size = len(system.coords)

    coords = torch.tensor(
        system.coords.reshape(stack_depth, system_size, 3),
        dtype=torch.float,
        device=device,
    ).requires_grad_(requires_grad)

    return dict(coords=coords, stack_depth=stack_depth, system_size=system_size)


@KinematicAtomicCoordinateProvider.factory_for.register(PackedResidueSystem)
@validate_args
def system_torsion_graph_inputs(
    system: PackedResidueSystem, device: torch.device, requires_grad: bool = True, **_
):
    """Constructor parameters for torsion space scoring.

    Extract constructor kwargs to initialize a `KinematicAtomicCoordinateProvider` and
    `BondedAtomScoreGraph` subclass supporting torsion-space scoring. This
    includes only `bond_torsion` dofs, a subset of valid kinematic dofs for the
    system.
    """

    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(system.bonds, system.torsion_metadata)

    # Select torsion dofs
    torsion_dofs = sys_kin.dof_metadata[
        (sys_kin.dof_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    # Extract kinematic-derived coordinates
    kincoords = sys_kin.extract_kincoords(system.coords).to(device)

    # Initialize op for torsion-space kinematics
    kinop = KinematicOp.from_coords(sys_kin.kintree, torsion_dofs, kincoords)

    return dict(
        dofs=kinop.src_mobile_dofs.clone().requires_grad_(requires_grad), kinop=kinop
    )
