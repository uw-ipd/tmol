"""Cartesian-bonded parameter database.

Per-residue length / angle / torsion / improper / hydroxyl-torsion
parameter groups. Straightforward port of
:mod:`tmol.database.scoring.cartbonded`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import attr
import yaml

from ._structure import structure


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
    residue_params: Dict[str, CartRes]

    @classmethod
    def from_file(cls, path: str) -> "CartBondedDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return structure(raw, cls)


__all__ = [
    "LengthGroup",
    "AngleGroup",
    "TorsionGroup",
    "ImproperGroup",
    "HxlTorsionGroup",
    "CartRes",
    "CartBondedDatabase",
]
