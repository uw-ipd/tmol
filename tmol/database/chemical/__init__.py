from typing import Tuple, Optional

import attr
import cattr

import os
import yaml
import properties


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


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
