"""Hydrogen-bond parameter database.

Port of :mod:`tmol.database.scoring.hbond`. Upstream used a pandas
``DataFrame`` to provide an indexed donor/acceptor-type mapper; we
instead expose two plain Python dicts (``don_type_map`` /
``acc_type_map``) which is what the downstream code actually needs and
which are trivially JIT-friendly (static aux data).
"""

from __future__ import annotations

from typing import Dict, Tuple

import attr
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondGlobalParams:
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
class _HBondDatabaseRaw:
    global_parameters: HBondGlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    donor_type_params: Tuple[DonorTypeParam, ...]
    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    acceptor_type_params: Tuple[AcceptorTypeParam, ...]
    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: HBondGlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    don_type_params: Tuple[DonorTypeParam, ...]
    acc_type_params: Tuple[AcceptorTypeParam, ...]
    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]
    # Pure-Python alternatives to the upstream pandas index mappers:
    # {atom_type_name: donor_type_name}
    don_type_map: Dict[str, str]
    acc_type_map: Dict[str, str]

    @classmethod
    def from_file(cls, path: str) -> "HBondDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        rawdb = structure(raw, _HBondDatabaseRaw)
        return cls(
            global_parameters=rawdb.global_parameters,
            donor_atom_types=rawdb.donor_atom_types,
            acceptor_atom_types=rawdb.acceptor_atom_types,
            don_type_params=rawdb.donor_type_params,
            acc_type_params=rawdb.acceptor_type_params,
            pair_parameters=rawdb.pair_parameters,
            polynomial_parameters=rawdb.polynomial_parameters,
            don_type_map={x.d: x.donor_type for x in rawdb.donor_atom_types},
            acc_type_map={x.a: x.acceptor_type for x in rawdb.acceptor_atom_types},
        )


# Aliases consistent with the task spec.
GlobalParams = HBondGlobalParams


__all__ = [
    "HBondGlobalParams",
    "GlobalParams",
    "DonorAtomType",
    "AcceptorAtomType",
    "DonorTypeParam",
    "AcceptorTypeParam",
    "PolynomialParameters",
    "PairParameters",
    "HBondDatabase",
]
