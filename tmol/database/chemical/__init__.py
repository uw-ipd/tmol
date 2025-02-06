from typing import Tuple, Optional, NewType
import pprint
import copy
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
    name: str = attr.ib()
    atom_type: str = attr.ib()


@attr.s(frozen=True, slots=True)
class AtomAlias:
    name: str = attr.ib()
    alt_name: str = attr.ib()


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
    io_equiv_class: str
    atoms: Tuple[Atom, ...]
    atom_aliases: Tuple[AtomAlias, ...]
    bonds: Tuple[Tuple[str, str], ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]
    properties: ChemicalProperties
    chi_samples: Tuple[ChiSamples, ...]
    default_jump_connection_atom: str

    def atom_name(self, index):
        return self.atoms[index].name


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IcoorVariant:
    name: str
    source: Optional[str] = None
    phi: Optional[DihedralAngle] = 0.0
    theta: Optional[BondAngle] = 0.0
    d: Optional[float] = 0.0
    parent: Optional[str] = None
    grand_parent: Optional[str] = None
    great_grand_parent: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerPropertiesVariant:
    polymer_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalPropertiesVariant:
    polymer: Optional[PolymerPropertiesVariant] = None


@attr.s(auto_attribs=True)
class VariantType:
    name: str
    display_name: str
    pattern: str
    remove_atoms: Tuple[str, ...]
    add_atoms: Tuple[Atom, ...]
    add_atom_aliases: Tuple[AtomAlias, ...]
    modify_atoms: Tuple[Atom, ...]
    add_connections: Tuple[Connection, ...]
    add_bonds: Tuple[Tuple[str, str], ...]
    icoors: Tuple[IcoorVariant, ...]


@attr.s(auto_attribs=True, frozen=True, slots=False)
class ChemicalDatabase:
    __default = None

    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RawResidueType, ...]
    variants: Tuple[VariantType, ...]

    # fd I don't particularly like this way of providing access to the unpatched database,
    # fd    but I'm not sure how else to do this.  Maybe keeping the unpatched in
    # fd    the default DB alongside the patched?
    @classmethod
    def get_default(cls) -> "ChemicalDatabase":
        """Load and return default parameter database."""
        if cls.__default is None:
            """cls.__default = ChemicalDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "..", "default", "chemical")
            )"""
            default = ChemicalDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "..", "default", "chemical")
            )
            extension = ChemicalDatabase.from_file(
                os.path.join(
                    os.path.dirname(__file__), "..", "default", "chemical", "extension"
                )
            )
            cls.__default = cls.concat_databases(default, extension)
        # print(cls.__default)
        # pprint.pprint(cls.__default, width=1)
        return cls.__default

    @classmethod
    def from_file(cls, path):
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)

        return cattr.structure(raw, cls)

    @classmethod
    def concat_databases(cls, db1, db2):
        element_types = db1.element_types + db2.element_types
        atom_types = db1.atom_types + db2.atom_types
        residues = db1.residues + db2.residues
        variants = db1.variants + db2.variants

        new_db = ChemicalDatabase(element_types, atom_types, residues, variants)

        return new_db
