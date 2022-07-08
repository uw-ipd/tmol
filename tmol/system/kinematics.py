from typing import Tuple, Union
import attr

import torch
import numpy

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.metadata import DOFMetadata
from tmol.kinematics.datatypes import KinForest

from .datatypes import torsion_metadata_dtype


@attr.s(auto_attribs=True, slots=True, frozen=True)
class KinematicDescription:
    """A kinematic tree paired and mobile dofs for the tree."""

    kinforest: KinForest
    dof_metadata: DOFMetadata

    @classmethod
    @validate_args
    def for_system(
        cls,
        system_size: int,
        bonds: Union[NDArray[int][:, 3], NDArray[int][:, 2]],
        torsion_metadata: Tuple[NDArray[torsion_metadata_dtype][:], ...],
    ):
        """Generate kinematics for system atoms and named torsions.

        Generate a kinematic tree fully spanning the atoms and named torsions
        within the system. ``KinForest.id`` is set of the atom index of the
        coordinate. Note that this is a covering of indices within the system
        "non-atom" ids are not present in the tree.
        """

        if bonds.shape[1] == 2:
            # right-pad with zeros to denote all bonds within a single system
            bonds = numpy.concatenate(
                (numpy.zeros((bonds.shape[0], 1), dtype=int), bonds), axis=1
            )

        # root all the trees in the kinforest at the first atom in each system.
        # use max on the stack index to count how many stacks there are
        roots = system_size * numpy.arange(1 + bonds[:, 0].max(), dtype=int)

        # construct torsion_bonds by merging the list of b-c atoms from each
        # torsion_metadata list and marking each one by the system from
        # which it originates.
        torsion_bonds_list = []
        for tmet in torsion_metadata:
            torsion_pairs = numpy.block(
                [[tmet["atom_index_b"]], [tmet["atom_index_c"]]]
            ).T
            torsion_bonds_list.append(
                torsion_pairs[numpy.all(torsion_pairs > 0, axis=-1)]
            )
        torsion_bonds = numpy.zeros(
            (sum(t.shape[0] for t in torsion_bonds_list), 3), dtype=int
        )
        start = numpy.cumsum(
            numpy.array([t.shape[0] for t in torsion_bonds_list], dtype=int)
        )
        for i, tbonds in enumerate(torsion_bonds_list):
            range = slice(0 if i == 0 else start[i - 1], start[i])
            torsion_bonds[range, 0] = i
            torsion_bonds[range, 1:3] = tbonds

        bonds = bonds[:, 0:1] * system_size + bonds[:, 1:3]
        torsion_bonds = torsion_bonds[:, 0:1] * system_size + torsion_bonds[:, 1:3]
        builder = KinematicBuilder().append_connected_components(
            roots,
            *KinematicBuilder.define_trees_with_prioritized_bonds(
                roots=roots,
                potential_bonds=bonds,
                prioritized_bonds=torsion_bonds,
            ),
            to_jump_nodes=numpy.array([], dtype=numpy.int32),
        )

        kinforest = builder.kinforest

        return cls(
            kinforest=builder.kinforest,
            dof_metadata=DOFMetadata.for_kinforest(kinforest),
        )

    def extract_kincoords(
        self, coords: NDArray[float][:, :, 3]
    ) -> Tensor[torch.double][:, 3]:
        """Extract the kinematic-derived coordinates from system coords.

        Extract kinematic-derived coordiantes, specified by kinforest.id,
        from the system coordinate buffere and set the proper global origin.
        """

        # Extract current state coordinates to render current dofs
        tcoords_flat = torch.from_numpy(coords).reshape(-1, 3)
        kincoords = tcoords_flat[self.kinforest.id.to(torch.long)]

        # Convert the -1 origin, a nan-coord, to zero
        assert self.kinforest.id[0] == -1
        assert torch.isnan(kincoords[0]).all()
        kincoords[0] = 0

        # Verify all kinematic coords are present
        if torch.isnan(kincoords[1:]).any():
            raise ValueError("kincoords dependent on nan coordinates")

        return kincoords
