import numpy
import torch

from typing import Optional

from ..types.functional import validate_args
from ..types.torch import Tensor

from ..kinematics.torch_op import KinematicOp
from ..kinematics.metadata import DOFTypes

from ..score.stacked_system import StackedSystem
from ..score.bonded_atom import BondedAtomScoreGraph
from ..score.rama.score_graph import RamaScoreGraph
from ..score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.database.scoring import RamaDatabase
from ..score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

# from ..score.residue_properties import ResidueProperties
# from ..score.torsions import AlphaAABackboneTorsionProvider
# from ..score.polymeric_bonds import PolymericBonds

from .packed import PackedResidueSystem
from .kinematics import KinematicDescription

from tmol.database import ParameterDatabase


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
    atom_names = system.atom_metadata["atom_name"].copy()[None, :]
    res_indices = system.atom_metadata["residue_index"].copy()[None, :]
    res_names = system.atom_metadata["residue_name"].copy()[None, :]

    if drop_missing_atoms:
        atom_types[0, numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        bonds=bonds,
        atom_types=atom_types,
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


@RamaScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def rama_graph_inputs(
    system: PackedResidueSystem,
    parameter_database: ParameterDatabase,
    rama_database: Optional[RamaDatabase] = None,
    **_,
):
    """Constructor parameters for rama scoring.

    Extract the atom indices of the 'phi' and 'psi' torsions
	from the torsion_metadata object, and the database.
    """
    if rama_database is None:
        rama_database = parameter_database.scoring.rama

    phis = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "phi"]
        ]
    )

    psis = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "psi"]
        ]
    )

    return dict(rama_database=rama_database, phis=phis, psis=psis)


@DunbrackScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def dunbrack_graph_inputs(
    system: PackedResidueSystem, parameter_database: ParameterDatabase, **_
):
    dun_phi = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "phi"]
        ],
        dtype=numpy.int64,
    )

    dun_psi = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "psi"]
        ],
        dtype=numpy.int64,
    )

    dun_chi1 = numpy.array(
        [
            [
                x["residue_index"],
                0,
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "chi1"]
        ],
        dtype=numpy.int64,
    )

    dun_chi2 = numpy.array(
        [
            [
                x["residue_index"],
                1,
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "chi2"]
        ],
        dtype=numpy.int64,
    )

    dun_chi3 = numpy.array(
        [
            [
                x["residue_index"],
                2,
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "chi3"]
        ],
        dtype=numpy.int64,
    )

    dun_chi4 = numpy.array(
        [
            [
                x["residue_index"],
                3,
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "chi4"]
        ],
        dtype=numpy.int64,
    )

    # print("dun_chi1.shape",dun_chi.shape)
    # print("dun_chi1.shape",dun_chi.shape)

    # merge the 4 chi tensors, sorting by residue index and chi index
    join_chi = numpy.concatenate((dun_chi1, dun_chi2, dun_chi3, dun_chi4), 0)
    chi_res = join_chi[:, 0]
    chi_inds = join_chi[:, 1]
    sort_inds = numpy.lexsort((chi_inds, chi_res))
    dun_chi = join_chi[sort_inds, :]

    print(dun_chi)

    return dict(
        dun_phi=torch.tensor(dun_phi, dtype=torch.long),
        dun_psi=torch.tensor(dun_psi, dtype=torch.long),
        dun_chi=torch.tensor(dun_chi, dtype=torch.long),
        dun_database=parameter_database.scoring.dun,
    )
