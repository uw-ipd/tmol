import attr

import torch
import numpy

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata
from tmol.kinematics.datatypes import KinTree

from .datatypes import torsion_metadata_dtype


@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinematicDescription:
    """A kinematic tree paired and mobile dofs for the tree."""
    kintree: KinTree
    dof_metadata: DOFMetadata

    @classmethod
    @validate_args
    def for_system(
            cls,
            bonds: NDArray(int)[:, 2],
            torsion_metadata: NDArray(torsion_metadata_dtype)[:],
    ):
        """Generate kinematics for system atoms and named torsions.

        Generate a kinematic tree fully spanning the atoms and named torsions
        within the system. ``KinTree.id`` is set of the atom index of the
        coordinate. Note that this is a covering of indices within the system
        "non-atom" ids are not present in the tree.
        """
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

    def extract_kincoords(
            self,
            coords: NDArray(float)[:, 3],
    ) -> Tensor(torch.double)[:, 3]:
        """Extract the kinematic-derived coordinates from system coords.

        Extract kinematic-derived coordiantes, specified by kintree.id,
        from the system coordinate buffere and set the proper global origin.
        """

        # Extract current state coordinates to render current dofs
        kincoords = torch.from_numpy(coords)[self.kintree.id]

        # Convert the -1 origin, a nan-coord, to zero
        assert self.kintree.id[0] == -1
        assert torch.isnan(kincoords[0]).all()
        kincoords[0] = 0

        # Verify all kinematic coords are present
        if torch.isnan(kincoords[1:]).any():
            raise ValueError("kincoords dependent on nan coordinates")

        return kincoords
