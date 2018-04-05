from typing import Sequence, Tuple

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
    atoms : Sequence[Atom]
    bonds : Sequence[Tuple[str, str]]
    lower_connect : str
    upper_connect : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase(properties.HasProperties):
    atom_types : Sequence[str]
    residues : Sequence[Residue]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        return cattr.structure(raw, cls)
