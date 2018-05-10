import numpy
import torch

from tmol.types.functional import validate_args

from tmol.kinematics.torch_op import KinematicOp
from tmol.kinematics.metadata import DOFTypes

from .packed import PackedResidueSystem
from .kinematics import SystemKinematics

from tmol.score.types import RealTensor


@validate_args
def system_real_space_graph_params(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
):
    bonds = system.bonds
    coords = (
        torch.from_numpy(system.coords).clone()
        .to(RealTensor.dtype)
        .requires_grad_(requires_grad)
    ) # yapf: disable

    atom_types = system.atom_metadata["atom_type"].copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        system_size=len(coords),
        bonds=bonds,
        coords=coords,
        atom_types=atom_types,
    )


@validate_args
def system_torsion_space_graph_params(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
):

    # Initialize kinematic tree for the system
    sys_kin = SystemKinematics.for_system(
        system.bonds, system.torsion_metadata
    )

    # Select torsion dofs
    torsion_dofs = sys_kin.dof_metadata[
        (sys_kin.dof_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    # Extract current state coordinates to render current dofs
    coords = torch.from_numpy(system.coords)
    kincoords = coords[sys_kin.kintree.id]

    # Global frame @ 0
    kincoords[0] = 0
    if torch.isnan(kincoords[1:]).any():
        raise ValueError("torsion space dofs do not support missing atoms")

    # Initialize op for torsion-space kinematics
    kop = KinematicOp.from_coords(
        sys_kin.kintree,
        torsion_dofs,
        kincoords,
    )

    # Bond/type data
    bonds = system.bonds
    atom_types = system.atom_metadata["atom_type"].copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        system_size=len(coords),
        dofs=kop.src_mobile_dofs.clone().requires_grad_(requires_grad),
        kinop=kop,
        bonds=bonds,
        atom_types=atom_types,
    )
