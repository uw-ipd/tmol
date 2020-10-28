from typing import Tuple, Optional, NewType
from tmol.utility.units import BondAngle, DihedralAngle

import attr
import cattr

import os
import yaml

AcceptorHybridization = NewType("AcceptorHybridization", str)
_acceptor_hybridizations = {"sp2", "sp3", "ring"}


def _parse_acceptor_hybridization(v, t):
    if v in _acceptor_hybridizations:
        return v
    else:
        raise ValueError(f"Invalid AcceptorHybridization value: {v}")


cattr.register_structure_hook(AcceptorHybridization, _parse_acceptor_hybridization)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Element:
    name: str
    atomic_number: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomType:
    name: str
    element: str
    is_acceptor: bool = False
    is_donor: bool = False
    is_hydroxyl: bool = False
    is_polarh: bool = False
    acceptor_hybridization: Optional[AcceptorHybridization] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    name: str
    atom_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    name: str
    phi: DihedralAngle
    theta: BondAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Connection:
    name: str
    atom: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class UnresolvedAtom:
    atom: Optional[str] = None
    connection: Optional[str] = None
    bond_sep_from_conn: Optional[int] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Torsion:
    name: str
    a: UnresolvedAtom
    b: UnresolvedAtom
    c: UnresolvedAtom
    d: UnresolvedAtom


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChiSamples:
    chi_dihedral: str
    samples: Tuple[float, ...]
    expansions: Tuple[float, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SidechainBuilding:
    chi_samples: ChiSamples


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerProperties:
    is_polymer: bool
    polymer_type: str
    backbone_type: str
    mainchain_atoms: Optional[Tuple[str, ...]]
    sidechain_chirality: str
    termini_variants: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ProtonationProperties:
    protonated_atoms: Tuple[str, ...]
    protonation_state: str
    pH: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalProperties:
    is_canonical: bool
    polymer: PolymerProperties
    chemical_modifications: Tuple[str, ...]
    connectivity: Tuple[str, ...]
    protonation: ProtonationProperties
    virtual: Tuple[str, ...]


@attr.s(auto_attribs=True)
class RawResidueType:
    name: str
    base_name: str
    name3: str
    atoms: Tuple[Atom, ...]
    bonds: Tuple[Tuple[str, str], ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]
    properties: ChemicalProperties
    # sidechain_building: Tuple[SidechainBuilding, ...]
    chi_samples: Tuple[ChiSamples, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RawResidueType, ...]

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)

        return cattr.structure(raw, cls)
