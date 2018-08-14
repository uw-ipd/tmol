from frozendict import frozendict
from toolz.curried import concat, map, compose
from typing import Mapping, Optional, NewType
import attr

import numpy

import tmol.database.chemical

AtomIndex = NewType("AtomIndex", int)
ConnectionIndex = NewType("ConnectionIndex", int)


@attr.s(slots=True, frozen=True)
class ResidueType(tmol.database.chemical.Residue):
    atom_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @atom_to_idx.default
    def _setup_atom_to_idx(self):
        return frozendict((a.name, i) for i, a in enumerate(self.atoms))

    coord_dtype: numpy.dtype = attr.ib()

    @coord_dtype.default
    def _setup_coord_dtype(self):
        return numpy.dtype([(a.name, float, 3) for a in self.atoms])

    bond_indicies: numpy.ndarray = attr.ib()

    @bond_indicies.default
    def _setup_bond_indicies(self):
        bondi = compose(list, sorted, set, concat)(
            [(ai, bi), (bi, ai)]
            for ai, bi in map(map(self.atom_to_idx.get), self.bonds)
        )

        bond_array = numpy.array(bondi)
        bond_array.flags.writeable = False
        return bond_array

    connection_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @connection_to_idx.default
    def _setup_connection_to_idx(self):
        return frozendict((c.name, self.atom_to_idx[c.atom]) for c in self.connections)

    connection_to_cidx: Mapping[Optional[str], ConnectionIndex] = attr.ib()

    @connection_to_cidx.default
    def _setup_connection_to_cidx(self):
        centries = [(None, -1)] + [(c.name, i) for i, c in enumerate(self.connections)]
        return frozendict(centries)

    def _repr_pretty_(self, p, cycle):
        p.text(f"ResidueType(name={self.name},...)")


@attr.s(slots=True, frozen=True)
class Residue:
    residue_type: ResidueType = attr.ib()
    coords: numpy.ndarray = attr.ib()

    @coords.default
    def _coord_buffer(self):
        return numpy.full((len(self.residue_type.atoms), 3), numpy.nan, dtype=float)

    @property
    def atom_coords(self) -> numpy.ndarray:
        return self.coords.reshape(-1).view(self.residue_type.coord_dtype)

    def attach_to(self, coord_buffer):
        assert coord_buffer.shape == self.coords.shape
        assert coord_buffer.dtype == self.coords.dtype

        coord_buffer[:] = self.coords

        return attr.evolve(self, coords=coord_buffer)

    def _repr_pretty_(self, p, cycle):
        p.text("Residue")
        with p.group(1, "(", ")"):
            p.text("residue_type=")
            p.pretty(self.residue_type)
            p.text(", coords=")
            p.break_()
            p.pretty(self.coords)
