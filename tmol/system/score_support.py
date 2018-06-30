import numpy
import torch

from ..types.functional import validate_args

from ..kinematics.torch_op import KinematicOp, ExecutionStrategy
from ..kinematics.metadata import DOFTypes

from ..score import (
    BondedAtomScoreGraph,
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from .packed import PackedResidueSystem
from .kinematics import KinematicDescription


@BondedAtomScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def bonded_atoms_for_system(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        **_,
):
    bonds = system.bonds

    atom_types = system.atom_metadata["atom_type"].copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        bonds=bonds,
        atom_types=atom_types,
    )


@CartesianAtomicCoordinateProvider.factory_for.register(PackedResidueSystem)
@validate_args
def coords_for_system(
        system: PackedResidueSystem,
        device: torch.device,
        requires_grad: bool = True,
        **_,
):
    """Extract constructor kwargs to initialize a `CartesianAtomicCoordinateProvider`"""

    coords = (
        torch.tensor(
            system.coords,
            dtype=torch.float,
            device=device,
        ).requires_grad_(requires_grad)
    )

    return dict(coords=coords, )


@KinematicAtomicCoordinateProvider.factory_for.register(PackedResidueSystem)
@validate_args
def system_torsion_graph_inputs(
        system: PackedResidueSystem,
        device: torch.device,
        requires_grad: bool = True,
        kinop_execution_strategy: ExecutionStrategy = ExecutionStrategy.
        default,
        **_,
):
    """Constructor parameters for torsion space scoring.

    Extract constructor kwargs to initialize a `KinematicAtomicCoordinateProvider` and
    `BondedAtomScoreGraph` subclass supporting torsion-space scoring. This
    includes only `bond_torsion` dofs, a subset of valid kinematic dofs for the
    system.
    """

    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(
        system.bonds, system.torsion_metadata
    )

    # Select torsion dofs
    torsion_dofs = sys_kin.dof_metadata[
        (sys_kin.dof_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    # Extract kinematic-derived coordinates
    kincoords = sys_kin.extract_kincoords(system.coords).to(device)

    # Initialize op for torsion-space kinematics
    kinop = KinematicOp.from_coords(
        sys_kin.kintree,
        torsion_dofs,
        kincoords,
        device,
        execution_strategy=kinop_execution_strategy,
    )

    return dict(
        dofs=kinop.src_mobile_dofs.clone().requires_grad_(requires_grad),
        kinop=kinop,
    )
