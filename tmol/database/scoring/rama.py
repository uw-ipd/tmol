import attr
import cattr
import yaml
import json

from typing import Tuple


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaEntry:
    phi: float
    psi: float
    prob: float
    energy: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTable:
    aa_class: str
    phi_step: float
    psi_step: float
    prepro: bool
    entries: Tuple[RamaEntry, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    tables: Tuple[RamaTable, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = json.load(infile)
        return cattr.structure(raw, cls)
