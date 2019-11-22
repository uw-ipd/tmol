import attr

import numpy

from tmol.kinematics.metadata import DOFMetadata

from tmol.system.kinematics import KinematicDescription


def test_metadata_dataframe_smoke(ubq_system):
    tsys = ubq_system
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )

    restored = DOFMetadata.from_frame(tkin.dof_metadata.to_frame())

    for a in attr.fields(DOFMetadata):
        numpy.testing.assert_array_equal(
            getattr(tkin.dof_metadata, a.name), getattr(restored, a.name)
        )
