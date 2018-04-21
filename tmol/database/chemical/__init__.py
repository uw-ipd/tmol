from typing import Tuple, NewType
from tmol.utility.units import parse_angle, u
from toolz import curry

import attr
import cattr

import os
import yaml
import properties


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
class Residue:
    name: str
    name3: str
    atoms: Tuple[Atom, ...]
    bonds: Tuple[Tuple[str, str], ...]
    lower_connect: str
    upper_connect: str
    icoors: Tuple[Icoor, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase(properties.HasProperties):
    atom_types: Tuple[str, ...]
    residues: Tuple[Residue, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        return cattr.structure(raw, cls)
