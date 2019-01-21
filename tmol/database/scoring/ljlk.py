import yaml

import attr
import cattr

from typing import Tuple, List

from enum import IntEnum


class Hyb(IntEnum):
    NONE = 0
    SP2 = 1
    SP3 = 2
    RING = 3


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKGlobalParameters:
    max_dis: float
    spline_start: float
    lj_hbond_OH_donor_dis: float
    lj_hbond_dis: float
    lj_hbond_hdis: float
    lj_switch_dis2sigma: float
    lk_min_dis2sigma: float
    lkb_water_dist: float
    lkb_water_angle_sp2: float
    lkb_water_angle_sp3: float
    lkb_water_angle_ring: float
    lkb_water_tors_sp2: List[float]
    lkb_water_tors_sp3: List[float]
    lkb_water_tors_ring: List[float]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKAtomTypeParameters:
    name: str
    elem: str
    lj_radius: float
    lj_wdepth: float
    lk_dgfree: float
    lk_lambda: float
    lk_volume: float
    hybridization: Hyb = 0
    is_acceptor: bool = False
    is_donor: bool = False
    is_hydroxyl: bool = False
    is_polarh: bool = False


@attr.s(auto_attribs=True, slots=True, frozen=True)
class LJLKDatabase:
    global_parameters: LJLKGlobalParameters
    atom_type_parameters: Tuple[LJLKAtomTypeParameters, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
