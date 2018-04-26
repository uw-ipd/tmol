import attr
import cattr
import yaml

from typing import Tuple


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DonorAtoms:
    d: str
    h: str
    donor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SP2AcceptorAtoms:
    a: str
    b: str
    b0: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SP3AcceptorAtoms:
    a: str
    b: str
    b0: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RingAcceptorAtoms:
    a: str
    b: str
    bp: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomGroups:
    donors: Tuple[DonorAtoms, ...]
    sp2_acceptors: Tuple[SP2AcceptorAtoms, ...]
    sp3_acceptors: Tuple[SP3AcceptorAtoms, ...]
    ring_acceptors: Tuple[RingAcceptorAtoms, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalTypes:
    donors: Tuple[str, ...]
    sp2_acceptors: Tuple[str, ...]
    sp3_acceptors: Tuple[str, ...]
    ring_acceptors: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    atom_groups: AtomGroups
    chemical_types: ChemicalTypes

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
