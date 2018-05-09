import attr

import numpy

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata


def test_metadata_dataframe_smoke(ubq_system):
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
    restored = DOFMetadata.from_frame(kinematic_metadata.to_frame())

    for a in attr.fields(DOFMetadata):
        numpy.testing.assert_array_equal(
            getattr(kinematic_metadata, a.name), getattr(restored, a.name)
        )
