from typing import Tuple, Optional, NewType
from tmol.utility.units import BondAngle, DihedralAngle

import attr
import cattr

import os
import yaml

AcceptorHybridization = NewType("AcceptorHybridization", str)
_acceptor_hybridizations = {"sp2", "sp3", "ring"}


def _parse_acceptor_hybridization(v, t):
    if v in _acceptor_hybridizations:
        return v
    else:
        raise ValueError(f"Invalid AcceptorHybridization value: {v}")


cattr.register_structure_hook(AcceptorHybridization, _parse_acceptor_hybridization)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomType:
    name: str
    element: str
    is_acceptor: bool = False
    is_donor: bool = False
    is_hydroxyl: bool = False
    is_polarh: bool = False
    acceptor_hybridization: Optional[AcceptorHybridization] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    name: str
    phi: DihedralAngle
    theta: BondAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Connection:
    name: str
    atom: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ConnectedAtom:
    atom: str
    connection: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Torsion:
    name: str
    a: ConnectedAtom
    b: ConnectedAtom
    c: ConnectedAtom
    d: ConnectedAtom


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Residue:
    name: str
    name3: str
    atoms: Tuple[Atom, ...]
    bonds: Tuple[Tuple[str, str], ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]
    hierarchies: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[Residue, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)

        return cattr.structure(raw, cls)
