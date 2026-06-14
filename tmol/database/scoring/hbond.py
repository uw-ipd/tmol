import attr
import cattr
import yaml
import pandas

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
    # Polynomial coefficients
    # The polynomial value is sum_{N=0..10} c_N * X^N. Default to 0
    c_0: float = 0.0
    c_1: float = 0.0
    c_2: float = 0.0
    c_3: float = 0.0
    c_4: float = 0.0
    c_5: float = 0.0
    c_6: float = 0.0
    c_7: float = 0.0
    c_8: float = 0.0
    c_9: float = 0.0
    c_10: float = 0.0


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PairParameters:
    donor_type: str
    acceptor_type: str
    AHdist: str
    cosBAH: str
    cosAHD: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabaseRaw:
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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: GlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    donor_type_params: Tuple[DonorTypeParam, ...]
    donor_type_mapper: pandas.DataFrame

    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    acceptor_type_params: Tuple[AcceptorTypeParam, ...]
    acceptor_type_mapper: pandas.DataFrame

    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]

    @classmethod
    def from_raw_hbond_db(cls, hbr):
        acc_ind = pandas.Index([x.a for x in hbr.acceptor_atom_types])
        don_ind = pandas.Index([x.d for x in hbr.donor_atom_types])

        atypes, acctypes = zip(
            *([(x.a, x.acceptor_type) for x in hbr.acceptor_atom_types])
        )
        dtypes, dontypes = zip(*([(x.d, x.donor_type) for x in hbr.donor_atom_types]))
        acc_df = pandas.DataFrame({"a": atypes, "acc_type": acctypes}, index=acc_ind)
        don_df = pandas.DataFrame({"d": dtypes, "don_type": dontypes}, index=don_ind)

        return cls(
            global_parameters=hbr.global_parameters,
            donor_atom_types=hbr.donor_atom_types,
            donor_type_params=hbr.donor_type_params,
            donor_type_mapper=don_df,
            acceptor_atom_types=hbr.acceptor_atom_types,
            acceptor_type_params=hbr.acceptor_type_params,
            acceptor_type_mapper=acc_df,
            pair_parameters=hbr.pair_parameters,
            polynomial_parameters=hbr.polynomial_parameters,
        )

    @classmethod
    def from_file(cls, path):
        hbr = HBondDatabaseRaw.from_file(path)
        return cls.from_raw_hbond_db(hbr)
