import attr
import cattr
import yaml

from typing import Tuple


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LengthGroup:
    res: str
    atm1: str
    atm2: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AngleGroup:
    res: str
    atm1: str
    atm2: str
    atm3: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TorsionGroup:
    res: str
    atm1: str
    atm2: str
    atm3: str
    atm4: str
    x0: float
    K: float
    period: int = 1


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HxlTorsionGroup:
    res: str
    atm1: str
    atm2: str
    atm3: str
    atm4: str
    k1: float = 0.0
    phi1: float = 0.0
    k2: float = 0.0
    phi2: float = 0.0
    k3: float = 0.0
    phi3: float = 0.0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CartBondedDatabase:
    length_parameters: Tuple[LengthGroup, ...]
    angle_parameters: Tuple[AngleGroup, ...]
    torsion_parameters: Tuple[TorsionGroup, ...]
    improper_parameters: Tuple[TorsionGroup, ...]
    hxltorsion_parameters: Tuple[HxlTorsionGroup, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
