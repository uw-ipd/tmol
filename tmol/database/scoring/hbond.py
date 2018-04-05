import attr
import cattr
import yaml

from typing import Tuple

@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondGlobalParams:
    max_dis : float

@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDonorAtoms:
    d : str
    h : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondAcceptorAtoms:
    a : str
    b : str
    b0 : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters : HBondGlobalParams
    donors : Tuple[HBondDonorAtoms, ...]
    acceptors : Tuple[HBondAcceptorAtoms, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
