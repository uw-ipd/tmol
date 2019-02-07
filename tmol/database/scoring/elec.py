import attr
import cattr
import yaml

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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ElecDatabase:
    global_parameters: GlobalParams
    atom_cp_reps_parameters: Tuple[CountPairReps, ...]
    atom_charge_parameters: Tuple[PartialCharges, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
