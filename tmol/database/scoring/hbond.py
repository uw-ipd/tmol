import attr
import cattr
import yaml

from typing import Tuple

@attr.s(auto_attribs=True, frozen=True, slots=True)
class GlobalParams:
    max_dis : float

@attr.s(auto_attribs=True, frozen=True, slots=True)
class DonorAtoms:
    d : str
    h : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class SP2AcceptorAtoms:
    a : str
    b : str
    b0 : str

@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters : GlobalParams
    donors : Tuple[DonorAtoms, ...]
    sp2_acceptors : Tuple[SP2AcceptorAtoms, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
