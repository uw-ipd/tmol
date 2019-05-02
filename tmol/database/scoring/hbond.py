import attr
import cattr
import yaml

from typing import Tuple


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GlobalParams:
    hb_sp2_range_span: float
    hb_sp2_BAH180_rise: float
    hb_sp2_outer_width: float
    hb_sp3_softmax_fade: float
    threshold_distance: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DonorAtomType:
    d: str
    donor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AcceptorAtomType:
    a: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DonorTypeParam:
    name: str
    weight: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AcceptorTypeParam:
    name: str
    weight: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PolynomialParameters:
    name: str
    dimension: str
    xmin: float
    xmax: float
    min_val: float
    max_val: float
    degree: int
    c_a: float
    c_b: float
    c_c: float
    c_d: float
    c_e: float
    c_f: float
    c_g: float
    c_h: float
    c_i: float
    c_j: float
    c_k: float


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PairParameters:
    donor_type: str
    acceptor_type: str
    AHdist: str
    cosBAH: str
    cosAHD: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: GlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    donor_type_params: Tuple[DonorTypeParam, ...]

    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    acceptor_type_params: Tuple[AcceptorTypeParam, ...]

    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        return cattr.structure(raw, cls)
