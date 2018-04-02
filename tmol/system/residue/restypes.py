from frozendict import frozendict
from toolz.curried import concat, map
from toolz import compose
import attr

import numpy

import collections

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
class AtomType(AttrMapping):
    name = attr.ib()
    atom_type = attr.ib()


@attr.s(slots=True, frozen=True)
class HBondDonor(AttrMapping):
    d = attr.ib()
    h = attr.ib()


@attr.s(slots=True, frozen=True)
class HBondRingAcceptor(AttrMapping):
    a = attr.ib()
    b = attr.ib()


@attr.s(slots=True, frozen=True)
class HBondSP2Acceptor(AttrMapping):
    a = attr.ib()
    b = attr.ib()
    b0 = attr.ib()


@attr.s(slots=True, frozen=True)
class HBondSP3Acceptor(AttrMapping):
    a = attr.ib()
    b = attr.ib()
    b0a = attr.ib()
    b0b = attr.ib()


@attr.s(slots=True, frozen=True)
class HBondAcceptorGroups(AttrMapping):
    ring = attr.ib(converter=compose(tuple, map(HBondRingAcceptor.from_dict)), default=[])
    sp2 = attr.ib(converter=compose(tuple, map(HBondSP2Acceptor.from_dict)), default=[])
    sp3 = attr.ib(converter=compose(tuple, map(HBondSP3Acceptor.from_dict)), default=[])


@attr.s(slots=True, frozen=True)
class HBondData(AttrMapping):
    donors = attr.ib(
        converter=compose(tuple, map(HBondDonor.from_dict)),
        default=[]
    )
    acceptors = attr.ib(
        converter=HBondAcceptorGroups.from_dict,
        default=HBondAcceptorGroups()
    )


@attr.s(slots=True, frozen=True)
class ResidueType(AttrMapping):
    name : str = attr.ib(converter=str)
    name3 : str = attr.ib(converter=str)
    atoms = attr.ib(converter=compose(tuple, map(AtomType.from_dict)))
    bonds = attr.ib(converter=compose(tuple, map(tuple)))

    lower_connect = attr.ib(converter=str)
    upper_connect = attr.ib(converter=str)

    hbond = attr.ib(converter=HBondData.from_dict, default=HBondData())

    atom_to_idx = attr.ib()
    @atom_to_idx.default
    def _setup_atom_to_idx(self):
        return frozendict(
            (a.name, i) for i, a in enumerate(self.atoms)
        )

    coord_dtype = attr.ib()
    @coord_dtype.default
    def _setup_coord_dtype(self):
        return numpy.dtype(
            [(a.name, float, 3) for a in self.atoms])

    bond_indicies = attr.ib()
    @bond_indicies.default
    def _setup_bond_indicies(self):
        bondi = list(sorted(set(concat(
            [(ai, bi), (bi, ai)]
            for ai, bi in map(map(self.atom_to_idx.get), self.bonds)
        ))))

        bond_array = numpy.array(bondi)
        bond_array.flags.writeable = False
        return bond_array

    lower_connect_idx = attr.ib()
    @lower_connect_idx.default
    def _setup_lower_connect_idx(self):
        return self.atom_to_idx[self.lower_connect]

    upper_connect_idx = attr.ib()
    @upper_connect_idx.default
    def _setup_upper_connect_idx(self):
        return self.atom_to_idx[self.upper_connect]

    def _repr_pretty_(self, p, cycle):
        p.text(f'ResidueType(name={self.name},...)')
