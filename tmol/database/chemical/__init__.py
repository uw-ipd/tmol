from typing import Tuple

import attr
import cattr

import os
import yaml
import properties

@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name : str
    atom_type : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class Residue:
    name  : str
    name3 : str
    atoms : Tuple[Atom, ...]
    bonds : Tuple[Tuple[str, str], ...]
    lower_connect : str
    upper_connect : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase(properties.HasProperties):
    atom_types : Tuple[str, ...]
    residues : Tuple[Residue, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        return cattr.structure(raw, cls)
