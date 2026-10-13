"""LJ / LK parameter database.

Mirrors :mod:`tmol.database.scoring.ljlk` but without ``cattr`` /
``torch``. Angle-valued globals (``lkb_water_angle_*`` and
``lkb_water_tors_*``) are parsed via the shared :data:`Angle` scalar
hook defined in :mod:`._structure`.
"""

from __future__ import annotations

from typing import Tuple

import attr
import yaml

from ._structure import Angle, structure


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParameters:
    max_dis: float
    lj_dlin_sigma_factor: float
    lj_hbond_OH_donor_dis: float
    lj_hbond_dis: float
    lj_hbond_hdis: float
    lk_min_dis2sigma: float
    lkb_water_dist: float
    lkb_water_angle_sp2: Angle
    lkb_water_angle_sp3: Angle
    lkb_water_angle_ring: Angle
    lkb_water_tors_sp2: Tuple[Angle, ...]
    lkb_water_tors_sp3: Tuple[Angle, ...]
    lkb_water_tors_ring: Tuple[Angle, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKAtomTypeParameters:
    name: str
    lj_radius: float
    lj_wdepth: float
    lk_dgfree: float
    lk_lambda: float
    lk_volume: float


# Alias that matches the name used in the task spec.
AtomLJLKParam = LJLKAtomTypeParameters


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKDatabase:
    global_parameters: LJLKGlobalParameters
    atom_type_parameters: Tuple[LJLKAtomTypeParameters, ...]

    @classmethod
    def from_file(cls, path: str) -> "LJLKDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return structure(raw, cls)


# Alias to match upstream spelling ``LJLKGlobal``.
LJLKGlobal = LJLKGlobalParameters


__all__ = [
    "LJLKGlobalParameters",
    "LJLKGlobal",
    "LJLKAtomTypeParameters",
    "AtomLJLKParam",
    "LJLKDatabase",
]
