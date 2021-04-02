import numpy
import torch

from collections import namedtuple
from typing import Optional

from ..types.functional import validate_args

from ..kinematics.operations import inverseKin

from ..score.stacked_system import StackedSystem
from ..score.bonded_atom import BondedAtomScoreGraph
from ..score.rama.score_graph import RamaScoreGraph
from ..score.omega.score_graph import OmegaScoreGraph
from ..score.dunbrack.score_graph import DunbrackScoreGraph
from tmol.database.scoring import RamaDatabase
from ..score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from .packed import PackedResidueSystem, PackedResidueSystemStack
from .kinematics import KinematicDescription

from tmol.database import ParameterDatabase


@StackedSystem.factory_for.register(PackedResidueSystem)
@validate_args
def stack_params_for_system(system: PackedResidueSystem, **_):
    return dict(stack_depth=1, system_size=int(system.system_size))


@StackedSystem.factory_for.register(PackedResidueSystemStack)
@validate_args
def stack_params_for_stacked_system(stack: PackedResidueSystemStack, **_):
    return dict(
        stack_depth=len(stack.systems),
        system_size=max(int(system.system_size) for system in stack.systems),
    )


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


@BondedAtomScoreGraph.factory_for.register(PackedResidueSystemStack)
@validate_args
def stacked_bonded_atoms_for_system(
    stack: PackedResidueSystemStack,
    stack_depth: int,
    system_size: int,
    drop_missing_atoms: bool = False,
    **_,
):
    bonds_for_systems = [
        bonded_atoms_for_system(sys, drop_missing_atoms) for sys in stack.systems
    ]

    for i, d in enumerate(bonds_for_systems):
        d["bonds"][:, 0] = i
    bonds = numpy.concatenate(tuple(d["bonds"] for d in bonds_for_systems))

    def expand_atoms(atdat):
        atdat2 = numpy.full((1, system_size), None, dtype=object)
        atdat2[0, : atdat.shape[1]] = atdat
        return atdat2

    def stackem(key):
        return numpy.concatenate([expand_atoms(d[key]) for d in bonds_for_systems])

    return dict(
        bonds=bonds,
        atom_types=stackem("atom_types"),
        atom_names=stackem("atom_names"),
        res_indices=stackem("res_indices"),
        res_names=stackem("res_names"),
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

    return dict(coords=coords)


@CartesianAtomicCoordinateProvider.factory_for.register(PackedResidueSystemStack)
@validate_args
def stacked_coords_for_system(
    stack: PackedResidueSystemStack,
    device: torch.device,
    stack_depth: int,
    system_size: int,
    requires_grad: bool = True,
    **_,
):
    """Extract constructor kwargs to initialize a `CartesianAtomicCoordinateProvider`"""

    coords_for_systems = [
        coords_for_system(sys, device, requires_grad) for sys in stack.systems
    ]

    coords = torch.full(
        (stack_depth, system_size, 3), numpy.nan, dtype=torch.float, device=device
    )
    for i, d in enumerate(coords_for_systems):
        coords[i, : d["coords"].shape[1]] = d["coords"]

    coords = coords.requires_grad_(requires_grad)

    return dict(coords=coords)


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
    tkintree = sys_kin.kintree.to(device)
    tdofmetadata = sys_kin.dof_metadata.to(device)

    # compute dofs from xyzs
    kincoords = sys_kin.extract_kincoords(system.coords).to(device)
    bkin = inverseKin(tkintree, kincoords)

    # dof mask

    return dict(
        dofs=bkin.raw.clone().requires_grad_(requires_grad),
        kintree=tkintree,
        dofmetadata=tdofmetadata,
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

    return dict(
        rama_database=rama_database, allphis=phis[None, :], allpsis=psis[None, :]
    )


@RamaScoreGraph.factory_for.register(PackedResidueSystemStack)
@validate_args
def rama_graph_for_stack(
    system: PackedResidueSystemStack,
    parameter_database: ParameterDatabase,
    rama_database: Optional[RamaDatabase] = None,
    **_,
):
    params = [
        rama_graph_inputs(sys, parameter_database, rama_database)
        for sys in system.systems
    ]

    max_nres = max(d["allphis"].shape[1] for d in params)

    def expand(t):
        ext = numpy.full((1, max_nres, 5), -1, dtype=int)
        ext[0, : t.shape[1], :] = t[0]
        return ext

    def stackem(key):
        return numpy.concatenate([expand(d[key]) for d in params])

    return dict(
        rama_database=params[0]["rama_database"],
        allphis=stackem("allphis"),
        allpsis=stackem("allpsis"),
    )


@OmegaScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def omega_graph_inputs(system: PackedResidueSystem, **_):
    """Constructor parameters for omega scoring.

    Extract the atom indices of the 'omega' torsions
    from the torsion_metadata object.
    """

    omegas = numpy.array(
        [
            [x["atom_index_a"], x["atom_index_b"], x["atom_index_c"], x["atom_index_d"]]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "omega"]
        ]
    )

    return dict(allomegas=omegas[None, :])


@OmegaScoreGraph.factory_for.register(PackedResidueSystemStack)
@validate_args
def omega_graph_for_stack(system: PackedResidueSystemStack, **_):
    params = [omega_graph_inputs(sys) for sys in system.systems]

    max_omegas = max(d["allomegas"].shape[1] for d in params)

    def expand(t):
        ext = numpy.full((1, max_omegas, 4), -1, dtype=int)
        ext[0, : t.shape[1], :] = t
        return ext

    return dict(allomegas=numpy.concatenate([expand(d["allomegas"]) for d in params]))


PhiPsiChi = namedtuple("PhiPsiChi", ["phi", "psi", "chi"])


def get_dunbrack_phi_psi_chi(
    system: PackedResidueSystem, device: torch.device
) -> PhiPsiChi:
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
        dtype=numpy.int32,
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
        dtype=numpy.int32,
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
        dtype=numpy.int32,
    )
    # print("dun_chi1")
    # print(dun_chi1)

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
        dtype=numpy.int32,
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
        dtype=numpy.int32,
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
        dtype=numpy.int32,
    )

    # merge the 4 chi tensors, sorting by residue index and chi index
    join_chi = numpy.concatenate((dun_chi1, dun_chi2, dun_chi3, dun_chi4), 0)
    chi_res = join_chi[:, 0]
    chi_inds = join_chi[:, 1]
    sort_inds = numpy.lexsort((chi_inds, chi_res))
    dun_chi = join_chi[sort_inds, :]

    return PhiPsiChi(
        torch.tensor(dun_phi[None, :], dtype=torch.int32, device=device),
        torch.tensor(dun_psi[None, :], dtype=torch.int32, device=device),
        torch.tensor(dun_chi[None, :], dtype=torch.int32, device=device),
    )


@DunbrackScoreGraph.factory_for.register(PackedResidueSystem)
@validate_args
def dunbrack_graph_inputs(
    system: PackedResidueSystem,
    parameter_database: ParameterDatabase,
    device: torch.device,
    **_,
):
    dunbrack_phi_psi_chi = get_dunbrack_phi_psi_chi(system, device)

    return dict(
        dun_phi=dunbrack_phi_psi_chi.phi,
        dun_psi=dunbrack_phi_psi_chi.psi,
        dun_chi=dunbrack_phi_psi_chi.chi,
        dun_database=parameter_database.scoring.dun,
    )


def get_dunbrack_phi_psi_chi_for_stack(
    systemstack: PackedResidueSystemStack, device: torch.device
) -> PhiPsiChi:
    phi_psi_chis = [
        get_dunbrack_phi_psi_chi(sys, device) for sys in systemstack.systems
    ]

    max_nres = max(phi_psi_chi.phi.shape[1] for phi_psi_chi in phi_psi_chis)
    max_nchi = max(phi_psi_chi.chi.shape[1] for phi_psi_chi in phi_psi_chis)

    def expand_dihe(t, max_size):
        ext = torch.full(
            (1, max_size, t.shape[2]), -1, dtype=torch.int32, device=t.device
        )
        ext[0, : t.shape[1], :] = t[0]
        return ext

    phi_psi_chi = PhiPsiChi(
        torch.cat(
            [expand_dihe(phi_psi_chi.phi, max_nres) for phi_psi_chi in phi_psi_chis]
        ),
        torch.cat(
            [expand_dihe(phi_psi_chi.psi, max_nres) for phi_psi_chi in phi_psi_chis]
        ),
        torch.cat(
            [expand_dihe(phi_psi_chi.chi, max_nchi) for phi_psi_chi in phi_psi_chis]
        ),
    )

    return phi_psi_chi


@DunbrackScoreGraph.factory_for.register(PackedResidueSystemStack)
@validate_args
def dunbrack_graph_for_stack(
    systemstack: PackedResidueSystemStack,
    parameter_database: ParameterDatabase,
    device: torch.device,
    **_,
):
    phi_psi_chi = get_dunbrack_phi_psi_chi_for_stack(systemstack, device)

    return dict(
        dun_phi=phi_psi_chi.phi,
        dun_psi=phi_psi_chi.psi,
        dun_chi=phi_psi_chi.chi,
        dun_database=parameter_database.scoring.dun,
    )
