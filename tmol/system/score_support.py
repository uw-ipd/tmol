import numpy
import torch

from ..types.functional import validate_args
from ..types.torch import Tensor

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

from ..database.chemical import three_letter_to_aatype


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

    Each of the residues in the system is represented by the three atom-index arrays,
    but some entries are going to be listed with an index of -1. Either these atoms
    don't exist for a canonical AA at that residue or that residue is not a canonical AA.
    """

    phi_inds = inds_for_torsion(system, device, "phi")
    psi_inds = inds_for_torsion(system, device, "psi")
    omega_inds = inds_for_torsion(system, device, "omega")

    res_aas = torch.tensor([
        three_letter_to_aatype[res.residue_type.name3]
        if res.residue_type.name3 in three_letter_to_aatype else -1
        for res in system.residues
    ],
                           dtype=torch.long,
                           device=device)

    return dict(
        phi_inds=phi_inds,
        psi_inds=psi_inds,
        omega_inds=omega_inds,
        res_aas=res_aas,
    )


@validate_args
def inds_for_torsion(
        system: PackedResidueSystem, device: torch.device, torsion_name: str
) -> Tensor(torch.long)[:, 4]:
    inds = torch.full((len(system.residues), 4),
                      -1,
                      dtype=torch.long,
                      device=device)

    tor_data = system.torsion_metadata[system.torsion_metadata["name"] ==
                                       torsion_name]
    tor_res = tor_data["residue_index"]
    inds[tor_res, 0] = torch.tensor(
        tor_data["atom_index_a"], dtype=torch.long, device=device
    )
    inds[tor_res, 1] = torch.tensor(
        tor_data["atom_index_b"], dtype=torch.long, device=device
    )
    inds[tor_res, 2] = torch.tensor(
        tor_data["atom_index_c"], dtype=torch.long, device=device
    )
    inds[tor_res, 3] = torch.tensor(
        tor_data["atom_index_d"], dtype=torch.long, device=device
    )
    return inds
