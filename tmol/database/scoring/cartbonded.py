import attr
import cattr
import yaml

from typing import Tuple


# TODO: remove all the 'Old' versions when we remove the non-res-centric version of cartbonded.
@attr.s(auto_attribs=True, slots=True, frozen=True)
class LengthGroupOld:
    res: str
    atm1: str
    atm2: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AngleGroupOld:
    res: str
    atm1: str
    atm2: str
    atm3: str
    x0: float
    K: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TorsionGroupOld:
    res: str
    atm1: str
    atm2: str
    atm3: str
    atm4: str
    x0: float
    K: float
    period: int = 1


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HxlTorsionGroupOld:
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
    type: int = 0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CartBondedDatabaseOld:
    length_parameters: Tuple[LengthGroupOld, ...]
    angle_parameters: Tuple[AngleGroupOld, ...]
    torsion_parameters: Tuple[TorsionGroupOld, ...]
    improper_parameters: Tuple[TorsionGroupOld, ...]
    hxltorsion_parameters: Tuple[HxlTorsionGroupOld, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LengthGroup:
    atm1: str
    atm2: str
    x0: float
    K: float
    type: int = 0


@attr.s(auto_attribs=True, slots=True, frozen=True)
class AngleGroup:
    atm1: str
    atm2: str
    atm3: str
    x0: float
    K: float
    type: int = 1


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TorsionGroup:
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
    type: int = 2


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ImproperGroup(TorsionGroup):
    type: int = 3


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HxlTorsionGroup(TorsionGroup):
    type: int = 4


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CartRes:
    length_parameters: Tuple[LengthGroup, ...]
    angle_parameters: Tuple[AngleGroup, ...]
    torsion_parameters: Tuple[TorsionGroup, ...]
    improper_parameters: Tuple[ImproperGroup, ...]
    hxltorsion_parameters: Tuple[HxlTorsionGroup, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CartBondedDatabase:
    residue_params: dict[str, CartRes]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
