from frozendict import frozendict
from toolz.curried import concat, map, compose
from typing import Mapping, Tuple
import attr

import numpy

import collections

import tmol.database.chemical


class AttrMapping(collections.abc.Mapping):
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __getitem__(self, k):
        return getattr(self, k)

    def __iter__(self):
        return iter(self.__slots__)

    def __len__(self):
        return len(self.__slots__)


@attr.s(slots=True, frozen=True)
class ResidueType(tmol.database.chemical.Residue, AttrMapping):
    atom_to_idx: Mapping[str, int] = attr.ib()

    @atom_to_idx.default
    def _setup_atom_to_idx(self):
        return frozendict((a.name, i) for i, a in enumerate(self.atoms))


    qualified_atom_types : Tuple[str] = attr.ib()

    @qualified_atom_types.default
    def _setup_qualified_atom_types(self):
        return tuple("/".join((self.name, a.atom_type)) for a in self.atoms)

    coord_dtype: numpy.dtype = attr.ib()

    @coord_dtype.default
    def _setup_coord_dtype(self):
        return numpy.dtype([(a.name, float, 3) for a in self.atoms])

    bond_indicies: numpy.ndarray = attr.ib()

    @bond_indicies.default
    def _setup_bond_indicies(self):
        bondi = compose(list, sorted, set, concat)(
            [(ai, bi), (bi, ai)]
            for ai, bi in map(map(self.atom_to_idx.get), self.bonds))

        bond_array = numpy.array(bondi)
        bond_array.flags.writeable = False
        return bond_array

    lower_connect_idx: int = attr.ib()

    @lower_connect_idx.default
    def _setup_lower_connect_idx(self):
        return self.atom_to_idx[self.lower_connect]

    upper_connect_idx: int = attr.ib()

    @upper_connect_idx.default
    def _setup_upper_connect_idx(self):
        return self.atom_to_idx[self.upper_connect]

    def _repr_pretty_(self, p, cycle):
        p.text(f'ResidueType(name={self.name},...)')

    mainchain_inds : Tuple[ int ] = attr.ib()

    @mainchain_inds.default
    def _setup_mainchain_inds( self ) :
        return [ self.atom_to_idx[ atname ] for atname in self.mainchain ]

    cutbond_inds: Tuple[ Tuple[ int, int ], ... ] = attr.ib()

    @cutbond_inds.default
    def _setup_cutbond_inds( self ) :
        return [ [ self.atom_to_idx[ atname ] for atname in atname_pair ] for atname_pair in self.cutbond ]

    chi_inds : Tuple[ Tuple[ int, int, int, int ], ... ] = attr.ib()

    @chi_inds.default
    def _setup_chi_inds( self ):
        return [ tuple( self.atom_to_idx[ atname ] for atname in x ) for x in self.chi ]

@attr.s(slots=True, frozen=True)
class Residue:
    residue_type: ResidueType = attr.ib()
    coords: numpy.ndarray = attr.ib()

    @coords.default
    def _coord_buffer(self):
        return numpy.full(
            (len(self.residue_type.atoms), 3), numpy.nan, dtype=float)

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
            p.text("type=")
            p.pretty(self.residue_type)
            p.text(", coords=")
            p.break_()
            p.pretty(self.coords)
