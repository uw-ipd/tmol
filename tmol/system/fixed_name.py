import numpy
import scipy.sparse

import properties

import tmol.io.pdb_parsing as pdb_parsing
from tmol.utility.array import coordinate_array_to_atoms, atom_array_to_coordinates

from properties import List
from tmol.properties.array import Array
from tmol.properties import eq_by_is
from tmol.properties.reactive import derived_from

class FixedNamedAtomSystem(properties.HasProperties):
    atoms = ["N", "CA", "C", "O"]
    atom_types = ["Nbb", "CAbb", "CObb", "OCbb"]
    bonds = [
        (("N", "CA", 0)),
        (("CA", "C", 0)),
        (("C", "O", 0)),
        (("C", "N", 1))
    ]

    coords = Array(
        "Atomic coordinates",
        dtype="f4", cast="unsafe")[:,3]

    types = Array("Per-atom types.", dtype=object)[:]

    bond_graph = eq_by_is(properties.Instance(
        "Inter-atomic bonds",
        scipy.sparse.spmatrix)
    )


    @property
    def atom_coords(self):
        return coordinate_array_to_atoms(self.coords, self.atoms)

    def load_pdb(self, pdb):
        atoms = pdb_parsing.parse_pdb(pdb)
        atoms = atoms[atoms.apply(lambda r: r["atomn"] in self.atoms, axis=1)]
        for t in ("model", "chain"):
            assert atoms[t].nunique() == 1
        resi = atoms["resi"].unique()
        assert numpy.all(resi == numpy.arange(resi[0], resi[-1] + 1))

        atoms = atoms.set_index(["resi", "atomn"])

        atom_buffer = numpy.empty_like(
            resi, dtype=numpy.dtype([(n, "f4", 3) for n in self.atoms]))

        for i, ri in enumerate(resi):
            for a in atom_buffer.dtype.names:
                atom_buffer[i][a] = atoms.loc[ri, a][["x", "y", "z"]].values

        self.coords = atom_array_to_coordinates(atom_buffer)
        self.types = numpy.array(self.atom_types * len(atom_buffer))
        self.bond_graph = self.generate_bonds(len(atom_buffer))

        return self

    def generate_bonds(self, nres):
        natoms = len(self.atoms)
        aidx = {a:i for i, a in enumerate(self.atoms)}

        up_entries = []
        down_entries = []

        for froma, toa, resoffset in self.bonds:
            assert resoffset >= 0
            up_entries.append(
                (numpy.arange(nres - resoffset) * natoms) + aidx[froma]
            )

            down_entries.append(
                ((numpy.arange(nres - resoffset) + resoffset) * natoms) + aidx[toa]
            )

        up_entries = numpy.concatenate(up_entries)
        down_entries = numpy.concatenate(down_entries)

        i = numpy.concatenate((up_entries, down_entries))
        j = numpy.concatenate((down_entries, up_entries))
        k = numpy.full_like(i, True, dtype=bool)

        return scipy.sparse.coo_matrix(
            (k, (i, j)),
            shape = (natoms * nres, natoms * nres),
        )


    def to_pdb(self, b = None):
        atom_records = numpy.zeros(
            (len(self.atom_coords), len(self.atoms)),
            dtype=pdb_parsing.atom_record_dtype
        )

        atom_records["resn"] = "CEN"
        atom_records["chain"] = "X"
        atom_records["resi"] = numpy.arange(len(self.atom_coords)).reshape((-1, 1))

        for i, n in enumerate(self.atom_coords.dtype.names):
            atom_records[:,i]["atomn"] = n
            atom_records[:,i]["x"] = self.atom_coords[n][:,0]
            atom_records[:,i]["y"] = self.atom_coords[n][:,1]
            atom_records[:,i]["z"] = self.atom_coords[n][:,2]

        atom_records = atom_records.ravel()
        atom_records["atomi"] = numpy.arange(len(atom_records))
        if b is not None:
            atom_records["b"] = b

        return pdb_parsing.to_pdb(atom_records.ravel())

