from typing import Tuple, Optional, NewType
from tmol.utility.units import parse_angle, u
from toolz import curry
from frozendict import frozendict

import attr
import cattr

import os
import yaml
import enum


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


PhiAngle = NewType("PhiAngle", float)
ThetaAngle = NewType("ThetaAngle", float)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    name: str
    phi: PhiAngle
    theta: ThetaAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


parse_angle = curry(parse_angle)

parse_phi = parse_angle(lim=(u("-pi rad"), u("pi rad")))
cattr.register_structure_hook(PhiAngle, lambda v, t: parse_phi(v))

parse_theta = parse_angle(lim=(u("0 rad"), u("pi rad")))
cattr.register_structure_hook(ThetaAngle, lambda v, t: parse_theta(v))


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
    mainchain: bool
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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    atom_types: Tuple[str, ...]
    residues: Tuple[Residue, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        return cattr.structure(raw, cls)
