import os

import attr
import cattr
from tmol.database._yaml import safe_load

from typing import Tuple


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GlobalParams:
    elec_min_dis: float
    elec_max_dis: float
    elec_sigmoidal_die_D: float
    elec_sigmoidal_die_D0: float
    elec_sigmoidal_die_S: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CountPairReps:
    res: str
    atm_inner: str
    atm_outer: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PartialCharges:
    res: str
    atom: str
    charge: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ElecDatabase:
    global_parameters: GlobalParams
    atom_cp_reps_parameters: Tuple[CountPairReps, ...]
    atom_charge_parameters: Tuple[PartialCharges, ...]

    @classmethod
    def from_file(cls, path, generated=()):
        with open(path, "r") as infile:
            raw = safe_load(infile)
        for extra in generated:
            if not os.path.exists(extra):
                continue
            with open(extra, "r") as infile:
                more = safe_load(infile)
            for section in ("atom_charge_parameters", "atom_cp_reps_parameters"):
                raw[section].extend(more.get(section, []))
        return cattr.structure(raw, cls)
