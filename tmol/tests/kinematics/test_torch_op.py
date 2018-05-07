import torch
import numpy

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata, DOFTypes
from tmol.kinematics.torch_op import KinematicOp


def test_kinematic_torch_op_smoke(ubq_system):
    tsys = ubq_system

    torsion_pairs = numpy.block([
        [tsys.torsion_metadata["atom_index_b"]],
        [tsys.torsion_metadata["atom_index_c"]],
    ]).T
    torsion_bonds = torsion_pairs[numpy.all(torsion_pairs > 0, axis=-1)]

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.component_for_prioritized_bonds(
            root=0,
            mandatory_bonds=torsion_bonds,
            all_bonds=tsys.bonds,
        )
    ).kintree

    kinematic_metadata = DOFMetadata.for_kintree(kintree)

    torsion_dofs = kinematic_metadata[
        (kinematic_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    coords = torch.from_numpy(tsys.coords)

    kop = KinematicOp.from_src_coords(
        kintree,
        torsion_dofs,
        coords,
    )

    refold_coords = kop.apply(kop.src_mobile_dofs)

    numpy.testing.assert_allclose(coords, refold_coords)
