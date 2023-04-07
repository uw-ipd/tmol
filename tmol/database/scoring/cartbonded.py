import attr
import cattr
import yaml

import typing
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


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartRes:
    length_parameters: Tuple[LengthGroup, ...]
    angle_parameters: Tuple[AngleGroup, ...]
    torsion_parameters: Tuple[TorsionGroup, ...]
    improper_parameters: Tuple[TorsionGroup, ...]
    hxltorsion_parameters: Tuple[HxlTorsionGroup, ...]


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


# TODO: Replace the original db format with these new versions and remove the 'new' if we want to keep this change.
@attr.s(auto_attribs=True, slots=True, frozen=True)
class LengthGroupNew:
    atm1: str
    atm2: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AngleGroupNew:
    atm1: str
    atm2: str
    atm3: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TorsionGroupNew:
    atm1: str
    atm2: str
    atm3: str
    atm4: str
    x0: float
    K: float
    period: int = 1


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HxlTorsionGroupNew:
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


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartRes:
    length_parameters: Tuple[LengthGroupNew, ...]
    angle_parameters: Tuple[AngleGroupNew, ...]
    torsion_parameters: Tuple[TorsionGroupNew, ...]
    improper_parameters: typing.Optional[Tuple[TorsionGroupNew, ...]] = None
    hxltorsion_parameters: typing.Optional[Tuple[HxlTorsionGroupNew, ...]] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CartBondedDatabaseNew:
    cartbonded_residues: typing.Dict[str, CartRes]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        # print(raw)
        return cattr.structure(raw, cls)
