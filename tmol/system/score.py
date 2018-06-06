import numpy
import torch

from ..types.functional import validate_args

from ..kinematics.torch_op import KinematicOp
from ..kinematics.metadata import DOFTypes

from ..score import (
    TorchDevice,
    BondedAtomScoreGraph,
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from .packed import PackedResidueSystem
from .kinematics import KinematicDescription


@validate_args
def system_device_graph_inputs(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
):
    """Extract constructor kwargs to initialize a `TorchDevice`."""
    return {"device": device}


@validate_args
def system_bond_graph_inputs(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
):
    """Extract constructor kwargs to initialize a `BondedAtomScoreGraph`."""

    bonds = system.bonds

    atom_types = system.atom_metadata["atom_type"].copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return {"bonds": bonds, "atom_types": atom_types}


@validate_args
def system_cartesian_graph_inputs(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
):
    """Extract constructor kwargs to initialize a `CartesianAtomicCoordinateProvider`"""

    coords = (
        torch.tensor(
            system.coords,
            dtype=torch.float,
            device=device,
        ).requires_grad_(requires_grad)
    )

    return {"coords": coords, "system_size": len(coords)}


@validate_args
def system_torsion_graph_inputs(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
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
    kop = KinematicOp.from_coords(
        sys_kin.kintree,
        torsion_dofs,
        kincoords,
    )

    return dict(
        dofs=kop.src_mobile_dofs.clone().requires_grad_(requires_grad),
        kinop=kop,
        system_size=len(system.coords),
    )


graph_input_components = {
    TorchDevice: system_device_graph_inputs,
    BondedAtomScoreGraph: system_bond_graph_inputs,
    CartesianAtomicCoordinateProvider: system_cartesian_graph_inputs,
    KinematicAtomicCoordinateProvider: system_torsion_graph_inputs,
}


def extract_graph_parameters(
        graph_class: type,
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
):
    graph_inputs = {}

    for graph_component in graph_class.mro():
        input_component = graph_input_components.get(graph_component, None)
        if input_component:
            graph_inputs.update(
                input_component(
                    system,
                    drop_missing_atoms,
                    requires_grad,
                    device,
                )
            )

    return graph_inputs
