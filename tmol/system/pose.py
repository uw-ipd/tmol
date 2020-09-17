import attr
import cattr

import numpy

from typing import Sequence, Tuple

from tmol.types.array import NDArray

from .restypes import ResidueType
from .datatypes import (
    atom_metadata_dtype,
    torsion_metadata_dtype,
    connection_metadata_dtype,
    partial_atom_id_dtype,
)

from tmol.database.chemical import ChemicalDatabase


@attr.s(auto_attribs=True)
class PackedResidueTypes:
    active_residues: Sequence[ResidueType]

    max_n_atoms: int
    n_atoms: NDArray(int)[:]  # ntypes

    # atom_types: NDArray(int)[:, :] # ntypes x max_n_atoms

    # parent_atoms: NDArray(int)[:, :]

    # bond_separation: NDArray(int)[:, :, :] # ntypes x max_n_atoms x max_n_atoms
    # max_n_interblock_bonds: int
    # n_interblock_bonds: NDArray(int)[:]
    # atoms_for_interblock_bonds: NDArray(int)[:, :] # ntypes x max_n_interres_bonds

    # max_n_donors: int
    # n_donors: NDArray(int)[:]
    # hydrogen_donor: NDArray(int)[:, :]
    # heavyatom_donor: NDArray(int)[:, :]
    #
    # max_n_acceptors: int
    # n_acceptors: NDArray(int)[:]
    # acceptors: NDArray(int)[:, :]
    # acceptor_base: NDArray(int)[:, :]
    # acceptor_base2: NDArray(int)[:, :]
    #
    # max_n_mainchain_atoms: int
    # n_mainchain_atoms: NDArray(int)[:]
    # mainchain_torsion_atoms: NDArray(partial_atom_id_type)[:, :, 4]
    #
    # max_n_chi: int
    # n_chi: NDArray(int)[:]
    # chi_torsion_atoms: NDArray(partial_atom_id_type)[:, :, 4]

    @property
    def n_types(self):
        return len(self.active_residues)

    @classmethod
    def from_restype_list(
        cls, active_residues: Sequence[ResidueType], chem_db: ChemicalDatabase
    ):
        max_n_atoms = cls.count_max_n_atoms(active_residues)
        n_atoms = cls.count_n_atoms(active_residues)

        # parent_atoms = cls.get_parent_atoms(active_residues, max_n_atoms)

        return cls(
            active_residues=active_residues, max_n_atoms=max_n_atoms, n_atoms=n_atoms
        )

    @classmethod
    def count_max_n_atoms(cls, active_residues: Sequence[ResidueType]):
        return max(len(restype.atoms) for restype in active_residues)

    @classmethod
    def count_n_atoms(cls, active_residues: Sequence[ResidueType]):
        return numpy.array(
            [len(restype.atoms) for restype in active_residues], dtype=numpy.int32
        )


@attr.s(auto_attribs=True)
class System:

    max_n_atoms: int

    coords: NDArray(float)[:, :, 3]
