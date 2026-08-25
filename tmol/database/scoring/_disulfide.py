import yaml

import attr
import cattr


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
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
