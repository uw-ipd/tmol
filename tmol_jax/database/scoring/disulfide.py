"""Disulfide scoring-term parameter database.

Pure YAML-only: carries a single ``global_parameters`` struct with the
distance / angle / dihedral polynomial coefficients used by the
disulfide score term.
"""

from __future__ import annotations

import attr
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DisulfideGlobalParameters:
    d_location: float
    d_scale: float
    d_shape: float

    a_logA: float
    a_kappa: float
    a_mu: float

    dss_logA1: float
    dss_kappa1: float
    dss_mu1: float
    dss_logA2: float
    dss_kappa2: float
    dss_mu2: float

    dcs_logA1: float
    dcs_mu1: float
    dcs_kappa1: float
    dcs_logA2: float
    dcs_mu2: float
    dcs_kappa2: float
    dcs_logA3: float
    dcs_mu3: float
    dcs_kappa3: float

    wt_dih_ss: float
    wt_dih_cs: float
    wt_ang: float
    wt_len: float
    shift: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DisulfideDatabase:
    global_parameters: DisulfideGlobalParameters

    @classmethod
    def from_file(cls, path: str) -> "DisulfideDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return structure(raw, cls)


__all__ = ["DisulfideGlobalParameters", "DisulfideDatabase"]
