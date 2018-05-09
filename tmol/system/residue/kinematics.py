import attr
import numpy

from tmol.types.array import NDArray
from tmol.types.functional import validate_args

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata
from tmol.kinematics.datatypes import KinTree

from .datatypes import torsion_metadata_dtype


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SystemKinematics:
    kintree: KinTree
    dof_metadata: DOFMetadata

    @classmethod
    @validate_args
    def for_system(
            cls,
            bonds: NDArray(int)[:, 2],
            torsion_metadata: NDArray(torsion_metadata_dtype)[:],
    ):
        torsion_pairs = numpy.block([
            [torsion_metadata["atom_index_b"]],
            [torsion_metadata["atom_index_c"]],
        ]).T
        torsion_bonds = torsion_pairs[numpy.all(torsion_pairs > 0, axis=-1)]

        builder = KinematicBuilder().append_connected_component(
            *KinematicBuilder.component_for_prioritized_bonds(
                root=0,
                mandatory_bonds=torsion_bonds,
                all_bonds=bonds,
            )
        )

        kintree = builder.kintree

        return cls(
            kintree=builder.kintree,
            dof_metadata=DOFMetadata.for_kintree(kintree),
        )
