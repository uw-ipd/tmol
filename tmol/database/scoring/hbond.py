import attr
import cattr
import yaml

from typing import Tuple


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GlobalParams:
    max_dis: float
    hb_sp2_range_span: float
    hb_sp2_BAH180_rise: float
    hb_sp2_outer_width: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DonorAtoms:
    d: str
    h: str
    donor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SP2AcceptorAtoms:
    a: str
    b: str
    b0: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SP3AcceptorAtoms:
    a: str
    b: str
    b0: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RingAcceptorAtoms:
    a: str
    b: str
    bp: str
    acceptor_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomGroups:
    donors: Tuple[DonorAtoms, ...]
    sp2_acceptors: Tuple[SP2AcceptorAtoms, ...]
    sp3_acceptors: Tuple[SP3AcceptorAtoms, ...]
    ring_acceptors: Tuple[RingAcceptorAtoms, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalTypes:
    donors: Tuple[str, ...]
    sp2_acceptors: Tuple[str, ...]
    sp3_acceptors: Tuple[str, ...]
    ring_acceptors: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HbondWeights:
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
    don_chem_type: str
    acc_chem_type: str
    AHdist: str
    cosBAH: str
    cosAHD: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: GlobalParams
    atom_groups: AtomGroups
    chemical_types: ChemicalTypes
    don_weights: Tuple[HbondWeights, ...]
    acc_weights: Tuple[HbondWeights, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]
    pair_parameters: Tuple[PairParameters, ...]

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as infile:
            raw = yaml.load(infile)
        return cattr.structure(raw, cls)
