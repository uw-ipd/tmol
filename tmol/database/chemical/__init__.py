"""Typed schemas for TMol chemical parameter data."""

from typing import Any, NewType, Optional, Tuple
from tmol.utility import BondAngle, DihedralAngle

import attr
import cattr

import os
from tmol.database._yaml import safe_load

AcceptorHybridization = NewType("AcceptorHybridization", str)
_acceptor_hybridizations = {"sp2", "sp3", "ring"}


def _parse_acceptor_hybridization(v, t):
    if v in _acceptor_hybridizations:
        return v
    raise ValueError(f"Invalid AcceptorHybridization value: {v}")


cattr.register_structure_hook(AcceptorHybridization, _parse_acceptor_hybridization)


def normalize_bond_tuples(raw: Any) -> Any:  # noqa: C901
    """Normalize legacy 2-field bond entries to include bond order.

    Historically, some YAML snippets used ``[atom1, atom2]`` for bonds.
    The typed schema expects 3-tuples: ``(atom1, atom2, bond_type)``.
    This helper expands 2-field entries to use ``"SINGLE"`` as default.

    Handles both the top-level dict shape (``chemical.yaml``) and a
    flat list of residue/variant dicts.
    """
    if isinstance(raw, dict):
        for key in ("residues", "variants"):
            entries = raw.get(key)
            if isinstance(entries, list):
                normalize_bond_tuples(entries)
        return raw

    if not isinstance(raw, list):
        return raw

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        for field in ("bonds", "add_bonds"):
            bonds = entry.get(field)
            if not isinstance(bonds, list):
                continue

            normalized = []
            for bond in bonds:
                if isinstance(bond, (list, tuple)) and len(bond) == 2:
                    normalized.append([bond[0], bond[1], "SINGLE", False])
                elif isinstance(bond, (list, tuple)) and len(bond) == 3:
                    normalized.append([bond[0], bond[1], bond[2], False])
                else:
                    normalized.append(bond)
            entry[field] = normalized
    return raw


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Element:
    """Chemical element name and atomic number."""

    name: str
    atomic_number: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomType:
    """Chemical atom-type properties used by scoring terms."""

    name: str
    element: str
    is_acceptor: bool = False
    is_donor: bool = False
    is_hydroxyl: bool = False
    is_polarh: bool = False
    acceptor_hybridization: Optional[AcceptorHybridization] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    """Named residue atom and its chemical atom type."""

    name: str = attr.ib()
    atom_type: str = attr.ib()


@attr.s(frozen=True, slots=True)
class AtomAlias:
    """Alternative input name for a residue atom."""

    name: str = attr.ib()
    alt_name: str = attr.ib()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Icoor:
    """Internal-coordinate definition for one residue atom."""

    name: str
    phi: DihedralAngle
    theta: BondAngle
    d: float
    parent: str
    grand_parent: str
    great_grand_parent: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Connection:
    """Named inter-residue connection and its bond type."""

    name: str
    atom: str
    type: str = "SINGLE"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class UnresolvedAtom:
    """Atom reference resolved from a name or residue connection."""

    atom: Optional[str] = None
    connection: Optional[str] = None
    bond_sep_from_conn: Optional[int] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Torsion:
    """Named torsion defined by four unresolved atoms."""

    name: str
    a: UnresolvedAtom
    b: UnresolvedAtom
    c: UnresolvedAtom
    d: UnresolvedAtom


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChiSamples:
    """Discrete samples and expansions for one chi dihedral."""

    chi_dihedral: str
    samples: Tuple[float, ...]
    expansions: Tuple[float, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SidechainBuilding:
    """Side-chain construction data for one chi dihedral."""

    chi_samples: ChiSamples


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerProperties:
    """Polymer identity and connectivity metadata for a residue."""

    is_polymer: bool
    # None for a non-polymer
    polymer_type: Optional[str]
    backbone_type: Optional[str]
    mainchain_atoms: Optional[Tuple[str, ...]]
    sidechain_chirality: str
    termini_variants: Tuple[str, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ProtonationProperties:
    """Protonation-state metadata for a residue."""

    protonated_atoms: Tuple[str, ...]
    protonation_state: str
    pH: float


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalProperties:
    """Chemical classification metadata for a residue."""

    is_canonical: bool
    polymer: PolymerProperties
    chemical_modifications: Tuple[str, ...]
    connectivity: Tuple[str, ...]
    protonation: ProtonationProperties
    virtual: Tuple[str, ...]


@attr.s(auto_attribs=True)
class RawResidueType:
    """Unpatched residue definition loaded from the chemical database."""

    name: str
    base_name: str
    name3: str
    io_equiv_class: str
    atoms: Tuple[Atom, ...]
    atom_aliases: Tuple[AtomAlias, ...]
    bonds: Tuple[tuple, ...]
    connections: Tuple[Connection, ...]
    torsions: Tuple[Torsion, ...]
    icoors: Tuple[Icoor, ...]
    properties: ChemicalProperties
    chi_samples: Tuple[ChiSamples, ...]
    default_jump_connection_atom: str
    # True when this residue's hydrogens were regenerated by the autogen ligand
    # pipeline (SMILES -> re-protonate -> OpenBabel). Their names/coordinates do
    # not correspond to the input structure, so pose construction must rebuild
    # them rather than trust input H by name. False for canonical / params-file
    # residues, whose H are authoritative.
    hydrogens_regenerated: bool = False
    # True only for blocks produced by user-defined ligand fragmentation.
    is_ligand_fragment: bool = False
    # One-letter sequence code; unique only within a backbone type ...
    #   "a" is both DA and RA.
    one_letter_code: Optional[str] = None

    def atom_name(self, index: int) -> str:
        """Return the name of the atom at ``index``."""
        return self.atoms[index].name


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IcoorVariant:
    """Patch operation that adds or modifies an internal coordinate."""

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
    """Polymer-property changes made by a residue patch."""

    polymer_type: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalPropertiesVariant:
    """Chemical-property changes made by a residue patch."""

    polymer: Optional[PolymerPropertiesVariant] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class VariantScope:
    """Which base residue types a patch is allowed to match.

    An unset field places no restriction. Only properties that patching cannot
    alter may be named here, so that a patch can never see another patch's
    effects; mainchain_atoms, for one, does not qualify.
    """

    backbone_types: Optional[Tuple[str, ...]] = None
    base_names: Optional[Tuple[str, ...]] = None

    def matches(self, res: RawResidueType) -> bool:
        """Return whether this scope accepts ``res``."""
        if self.base_names is not None and res.base_name not in self.base_names:
            return False
        backbone = res.properties.polymer.backbone_type
        return self.backbone_types is None or backbone in self.backbone_types


@attr.s(auto_attribs=True, frozen=True, slots=True)
class VariantType:
    """Patch that transforms a base residue into a chemical variant."""

    name: str
    display_name: str
    pattern: str
    remove_atoms: Tuple[str, ...]
    add_atoms: Tuple[Atom, ...]
    add_atom_aliases: Tuple[AtomAlias, ...]
    modify_atoms: Tuple[Atom, ...]
    add_connections: Tuple[Connection, ...]
    add_bonds: Tuple[tuple, ...]
    icoors: Tuple[IcoorVariant, ...]
    add_torsions: Tuple[Torsion, ...] = ()
    add_chi_samples: Tuple[ChiSamples, ...] = ()
    applies_to: VariantScope = VariantScope()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ChemicalDatabase:
    """Immutable collection of chemical types, residues, and patches."""

    __default = None

    element_types: Tuple[Element, ...]
    atom_types: Tuple[AtomType, ...]
    residues: Tuple[RawResidueType, ...]
    variants: Tuple[VariantType, ...]

    @classmethod
    def get_default(cls) -> "ChemicalDatabase":
        """Load and return default parameter database."""
        if cls.__default is None:
            cls.__default = ChemicalDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "..", "default", "chemical")
            )
        return cls.__default

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ChemicalDatabase":
        """Load a chemical database from a directory containing YAML data."""
        path = os.path.join(path, "chemical.yaml")
        with open(path, "r") as infile:
            raw = safe_load(infile)
        raw = normalize_bond_tuples(raw)

        return cattr.structure(raw, cls)
