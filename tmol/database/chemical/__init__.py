from typing import Tuple, Optional, NewType
from tmol.utility.units import parse_angle, u
from toolz import curry
from frozendict import frozendict

import attr
import cattr

import os
import yaml
import enum


class AAType(enum.IntEnum):
    aa_ala = 0
    aa_cys = enum.auto()
    aa_asp = enum.auto()
    aa_glu = enum.auto()
    aa_phe = enum.auto()
    aa_gly = enum.auto()
    aa_his = enum.auto()
    aa_ile = enum.auto()
    aa_lys = enum.auto()
    aa_leu = enum.auto()
    aa_met = enum.auto()
    aa_asn = enum.auto()
    aa_pro = enum.auto()
    aa_gln = enum.auto()
    aa_arg = enum.auto()
    aa_ser = enum.auto()
    aa_thr = enum.auto()
    aa_val = enum.auto()
    aa_trp = enum.auto()
    aa_tyr = enum.auto()


one_letter_to_aatype = frozendict({
    "A": AAType.aa_ala,
    "C": AAType.aa_cys,
    "D": AAType.aa_asp,
    "E": AAType.aa_glu,
    "F": AAType.aa_phe,
    "G": AAType.aa_gly,
    "H": AAType.aa_his,
    "I": AAType.aa_ile,
    "K": AAType.aa_lys,
    "L": AAType.aa_leu,
    "M": AAType.aa_met,
    "N": AAType.aa_asn,
    "P": AAType.aa_pro,
    "Q": AAType.aa_gln,
    "R": AAType.aa_arg,
    "S": AAType.aa_ser,
    "T": AAType.aa_thr,
    "V": AAType.aa_val,
    "W": AAType.aa_trp,
    "Y": AAType.aa_tyr
})

aatype_to_one_letter = ["X"] * len(AAType)
for key in one_letter_to_aatype:
    aatype_to_one_letter[one_letter_to_aatype[key]] = key

three_letter_to_aatype = frozendict({
    "ALA": AAType.aa_ala,
    "CYS": AAType.aa_cys,
    "ASP": AAType.aa_asp,
    "GLU": AAType.aa_glu,
    "PHE": AAType.aa_phe,
    "GLY": AAType.aa_gly,
    "HIS": AAType.aa_his,
    "ILE": AAType.aa_ile,
    "LYS": AAType.aa_lys,
    "LEU": AAType.aa_leu,
    "MET": AAType.aa_met,
    "ASN": AAType.aa_asn,
    "PRO": AAType.aa_pro,
    "GLN": AAType.aa_gln,
    "ARG": AAType.aa_arg,
    "SER": AAType.aa_ser,
    "THR": AAType.aa_thr,
    "VAL": AAType.aa_val,
    "TRP": AAType.aa_trp,
    "TYR": AAType.aa_tyr
})

aatype_to_three_letter = ["XXX"] * len(AAType)
for key in three_letter_to_aatype:
    aatype_to_three_letter[three_letter_to_aatype[key]] = key


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


PhiAngle = NewType("PhiAngle", float)
ThetaAngle = NewType("ThetaAngle", float)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    name: str
    phi: PhiAngle
    theta: ThetaAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


parse_angle = curry(parse_angle)

parse_phi = parse_angle(lim=(u("-pi rad"), u("pi rad")))
cattr.register_structure_hook(PhiAngle, lambda v, t: parse_phi(v))

parse_theta = parse_angle(lim=(u("0 rad"), u("pi rad")))
cattr.register_structure_hook(ThetaAngle, lambda v, t: parse_theta(v))


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Connection:
    name: str
    atom: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ConnectedAtom:
    atom: str
    connection: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Torsion:
    name: str
    mainchain: bool
    a: ConnectedAtom
    b: ConnectedAtom
    c: ConnectedAtom
    d: ConnectedAtom


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Residue:
    name: str
    name3: str
    atoms: Tuple[Atom, ...]
    bonds: Tuple[Tuple[str, str], ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    atom_types: Tuple[str, ...]
    residues: Tuple[Residue, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.load(infile)

        return cattr.structure(raw, cls)
