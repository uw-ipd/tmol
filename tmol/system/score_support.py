import numpy
import torch

from ..types.functional import validate_args

from ..kinematics.torch_op import KinematicOp
from ..kinematics.metadata import DOFTypes

from ..score import (
    BondedAtomScoreGraph,
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)
from ..score.torsions import AlphaAABackboneTorsionProvider

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
    kop = KinematicOp.from_coords(
        sys_kin.kintree,
        torsion_dofs,
        kincoords,
    )

    return dict(
        dofs=kop.src_mobile_dofs.clone().requires_grad_(requires_grad),
        kinop=kop,
    )


@AlphaAABackboneTorsionProvider.factory_for.register(PackedResidueSystem)
@validate_args
def system_torsions_from_coordinates(
        system: PackedResidueSystem,
        device: torch.device,
        **_,
):
    """Constructor for finding the named backbone torsions of residues in the system

    I don't really know what I'm doing with this code. I need phi and psi to calculate
    Rama, and the PackedResidueSystem contains the torsion metadata.
    I also need to be able to talk about the AA type for residue i and the AA type
    for residue i+1 (to know whether or not it is proline), and what worries me about
    this class is that I'm introducing a potentially brittle correspondence between
    the entries of the phi_inds, etc. arrays with a presumbably separate class
    that reports which AA each residue is using. (That is, if a residue type which
    is not a canonical AA also reports a dihedral named "phi", then there will
    be N+1 entries in the phi_inds tensor, but only N entries in the
    (hypothetical) aa-type tensor and it won't be clear which entries in one
    tensor go with which entries in the other one.
    """

    phi_inds = torch.full((len(system.residues), 4),
                          -1,
                          dtype=torch.long,
                          device=device)
    psi_inds = torch.full((len(system.residues), 4),
                          -1,
                          dtype=torch.long,
                          device=device)
    omega_inds = torch.full((len(system.residues), 4),
                            -1,
                            dtype=torch.long,
                            device=device)

    phi_data = system.torsion_metadata[system.torsion_metadata["name"] == "phi"
                                       ]
    phi_inds[:, 0] = torch.tensor(
        phi_data["atom_index_a"], dtype=torch.long, device=device
    )
    phi_inds[:, 1] = torch.tensor(
        phi_data["atom_index_b"], dtype=torch.long, device=device
    )
    phi_inds[:, 2] = torch.tensor(
        phi_data["atom_index_c"], dtype=torch.long, device=device
    )
    phi_inds[:, 3] = torch.tensor(
        phi_data["atom_index_d"], dtype=torch.long, device=device
    )

    psi_data = system.torsion_metadata[system.torsion_metadata["name"] == "psi"
                                       ]
    psi_inds[:, 0] = torch.tensor(
        psi_data["atom_index_a"], dtype=torch.long, device=device
    )
    psi_inds[:, 1] = torch.tensor(
        psi_data["atom_index_b"], dtype=torch.long, device=device
    )
    psi_inds[:, 2] = torch.tensor(
        psi_data["atom_index_c"], dtype=torch.long, device=device
    )
    psi_inds[:, 3] = torch.tensor(
        psi_data["atom_index_d"], dtype=torch.long, device=device
    )

    omega_data = system.torsion_metadata[system.torsion_metadata["name"] ==
                                         "omega"]
    omega_inds[:, 0] = torch.tensor(
        omega_data["atom_index_a"], dtype=torch.long, device=device
    )
    omega_inds[:, 1] = torch.tensor(
        omega_data["atom_index_b"], dtype=torch.long, device=device
    )
    omega_inds[:, 2] = torch.tensor(
        omega_data["atom_index_c"], dtype=torch.long, device=device
    )
    omega_inds[:, 3] = torch.tensor(
        omega_data["atom_index_d"], dtype=torch.long, device=device
    )

    return dict(
        phi_inds=phi_inds,
        psi_inds=psi_inds,
        omega_inds=omega_inds,
    )
