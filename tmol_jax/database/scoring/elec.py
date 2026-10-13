"""Electrostatics parameter database.

Port of :mod:`tmol.database.scoring.elec`. The YAML contains three
top-level sections: ``global_parameters`` (sigmoidal-dielectric
coefficients), ``atom_cp_reps_parameters`` (residue-scoped count-pair
representative mappings), and ``atom_charge_parameters`` (per-residue
per-atom partial charges).

The downstream tmol code also maintains a ``cp_reps_path_distance``
array, but that is *derived* from the chemical DB + the cp_reps
mapping, not stored in the YAML — it therefore isn't a field here.
"""

from __future__ import annotations

from typing import Tuple

import attr
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ElecGlobalParams:
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
    global_parameters: ElecGlobalParams
    atom_cp_reps_parameters: Tuple[CountPairReps, ...]
    atom_charge_parameters: Tuple[PartialCharges, ...]

    @classmethod
    def from_file(cls, path: str) -> "ElecDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return structure(raw, cls)


# Friendlier aliases that match the task-spec naming.
GlobalParams = ElecGlobalParams
atom_charges = PartialCharges  # noqa: N816 - exported for symmetry


__all__ = [
    "ElecGlobalParams",
    "GlobalParams",
    "CountPairReps",
    "PartialCharges",
    "ElecDatabase",
]
