from tmol.database._yaml import safe_load

import attr
import cattr
import math


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

    # Rosetta FullatomDisulfideParams13 mixed D/L distribution. Defaults keep
    # existing custom parameter files readable; means are stored in radians.
    dss_mixed_logA1: float = -28.1535
    dss_mixed_kappa1: float = 25.9429
    dss_mixed_mu1: float = -math.pi / 2
    dss_mixed_logA2: float = -28.1535
    dss_mixed_kappa2: float = 25.9429
    dss_mixed_mu2: float = math.pi / 2


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DisulfideDatabase:
    global_parameters: DisulfideGlobalParameters

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = safe_load(infile)
        return cattr.structure(raw, cls)
