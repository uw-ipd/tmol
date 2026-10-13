"""Torch-free chemical records matching TMol's canonical parameter schema."""

from __future__ import annotations

import os
import typing
from typing import Any, NewType, Optional, Tuple
from functools import lru_cache

import attr
import yaml
from tmol_jax.utility.angle import parse_angle as _parse_angle

BondAngle = NewType("BondAngle", float)
DihedralAngle = NewType("DihedralAngle", float)
AcceptorHybridization = NewType("AcceptorHybridization", str)
CoordinationGeometry = NewType("CoordinationGeometry", str)
_ACCEPTOR_HYBRIDIZATIONS = {"sp2", "sp3", "ring"}
GEOMETRY_SITE_COUNT = {
    "linear": 2,
    "trigonal": 3,
    "tetrahedral": 4,
    "square_planar": 4,
    "trigonal_bipyramidal": 5,
    "square_pyramidal": 5,
    "octahedral": 6,
    "irregular": None,
}
GENERATED_RESIDUE_FILES = (
    "d_amino_acids.yaml",
    "metal_ions.yaml",
    "metal_clusters.yaml",
)


def safe_load(stream):
    return yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


@lru_cache(None)
def _type_hints(cls):
    return typing.get_type_hints(cls)


def _parse_acceptor_hybridization(v: Any) -> Optional[str]:
    if v is None:
        return None
    if v in _ACCEPTOR_HYBRIDIZATIONS:
        return v
    raise ValueError(f"Invalid AcceptorHybridization value: {v!r}")


def _is_attrs_class(t: Any) -> bool:
    return isinstance(t, type) and hasattr(t, "__attrs_attrs__")


def _unwrap_optional(t: Any) -> Tuple[Any, bool]:
    """If ``t`` is ``Optional[X]`` / ``Union[X, None]``, return ``(X, True)``.

    Otherwise return ``(t, False)``.
    """
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0], True
    return t, False


def _structure(value: Any, target_type: Any) -> Any:
    """Recursively structure a YAML-decoded value into ``target_type``.

    Handles primitives, ``Optional[T]``, ``Tuple[T, ...]`` /
    ``Tuple[T1, T2]``, and nested attrs dataclasses. Custom hooks are
    applied for angle-like types and for the ``AcceptorHybridization``
    string enum.
    """

    # Unwrap Optional / Union[..., None]
    target_type, _optional = _unwrap_optional(target_type)

    if value is None:
        return None

    # Custom scalar hooks -------------------------------------------------
    if target_type is BondAngle or target_type is DihedralAngle:
        return _parse_angle(value)
    if target_type is AcceptorHybridization:
        return _parse_acceptor_hybridization(value)

    # Nested attrs dataclasses --------------------------------------------
    if _is_attrs_class(target_type):
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected mapping for {target_type.__name__}, got {type(value).__name__}"
            )
        kwargs = {}
        # Use resolved type hints so forward references (``"Foo"``) work.
        try:
            hints = _type_hints(target_type)
        except Exception:
            hints = {a.name: a.type for a in target_type.__attrs_attrs__}
        for field in target_type.__attrs_attrs__:
            if field.name in value:
                kwargs[field.name] = _structure(
                    value[field.name], hints.get(field.name, field.type)
                )
            elif field.default is not attr.NOTHING:
                # Leave it for attrs to apply the default.
                pass
            else:
                raise KeyError(
                    f"Missing required field {field.name!r} for {target_type.__name__}"
                )
        return target_type(**kwargs)

    # Generic container types ---------------------------------------------
    if target_type is tuple:
        return tuple(value)

    origin = typing.get_origin(target_type)
    args = typing.get_args(target_type)

    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(_structure(v, elem_t) for v in value)
        # Fixed-length tuple (e.g. ``Tuple[str, str]`` for a bond pair).
        if len(args) == len(value):
            return tuple(_structure(v, a) for v, a in zip(value, args))
        # Fall back to element-by-element using the first type arg.
        if args:
            return tuple(_structure(v, args[0]) for v in value)
        return tuple(value)

    if origin is list:
        elem_t = args[0] if args else Any
        return [_structure(v, elem_t) for v in value]

    # Plain primitives (str, int, float, bool, Any, or unknown).
    return value


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
    # Metals are typed per oxidation state, because the state sets both the
    # electrostatic charge and the Lennard-Jones radius: Fe2p is not Fe3p.
    is_metal: bool = False
    oxidation_state: Optional[int] = None
    # donates a lone pair to a metal; not is_acceptor, which follows Rosetta's hbond
    #    typing and excludes thiolate and thioether sulfur
    is_metal_donor: bool = False
    # the type this atom takes when a covalent bond replaces its hydrogen:
    #    a hydroxyl becomes an ether, a thiol a thioether
    conjugated_type: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Atom:
    """Named residue atom and its chemical atom type."""

    name: str = attr.ib()
    atom_type: str = attr.ib()
    # Generic bonded lookup can describe a changed local environment without
    # changing nonbonded typing or transferring Rosetta torsion ownership.
    #
    # This is deliberately per atom rather than a wider genbonded hierarchy.
    # The hierarchy is a property of an atom type and is global: it fixes gaps
    # like Nbb reaching NG2 without passing through Nad, and that fix belongs
    # in scoring/genbonded.yaml. Conjugation is different -- it changes one
    # atom's chemistry as a function of what became bonded to it, while the
    # base residue keeps the Rosetta type that owns its canonical torsions.
    # ASN:conj_ND2 is the worked example: ND2 stays Nglyc for the Rosetta
    # terms and looks up generic parameters as the amide Nad it has become.
    # No per-type hierarchy can express that, because it depends on the
    # partner, not on the atom type.
    genbonded_type: Optional[str] = None
    # Fallback peptide role for cross-residue Cartesian parameter lookup.
    # Authored atom names and explicitly supplied parameters retain priority.
    cartbonded_reference: Optional[str] = None


@attr.s(frozen=True, slots=True)
class AtomAlias:
    """Alternative input name for a residue atom."""

    name: str = attr.ib()
    alt_name: str = attr.ib()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Name3Alias:
    """An input residue name that is read as another residue's.

    The residue is read, scored and written as ``read_as``; its own name does
    not survive. Reserved for a component the database deliberately does not
    describe, where the substitute is chemically the same molecule.
    """

    name3: str
    read_as: str


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
    # Whether the bond this connection completes closes a ring. Generic bonded
    # lookup bins a bond by order *and* ring membership, and the two sets are
    # disjoint, so a connection that defaults to False can never match a rule
    # written for a ring bond. Only a cut that splits a ring sets this: a
    # polymer up/down bond and an ordinary attachment do not close one.
    in_ring: bool = False
    # False for a bond outside the kinematic tree and packing groups (a disulfide,
    #    a metal-ligand bond); it still counts for count-pair
    kinematic: bool = True


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
    """Discrete values a torsion is sampled at, where no library supplies them.

    A proton chi turns a hydrogen and is optH's to place as well as the
    packer's; any other sampled chi moves heavy atoms and is the packer's
    alone.
    """

    chi_dihedral: str
    samples: Tuple[float, ...]
    expansions: Tuple[float, ...]
    is_proton: bool = True


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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MetalSite:
    """The coordination sites on one metal atom of a residue type."""

    metal_atom: str
    geometry: CoordinationGeometry
    # in-block atoms filling sites (heme nitrogens, a cluster's bridging sulfurs,
    #    listed by every metal they bridge); detection fills only the rest
    internal_satisfiers: Tuple[str, ...] = ()
    # free-site directions, where lk_ball builds waters; a free ion's are placed
    #    when its sites are assigned
    site_virts: Tuple[str, ...] = ()
    # connections a donor fills, parallel to site_virts (an untemplated ion has
    #    no virtuals); an unfilled one is an open site
    site_connections: Tuple[str, ...] = ()

    @property
    def n_free_sites(self) -> Optional[int]:
        """Sites detection may fill; None when the geometry is untemplated."""
        n_total = GEOMETRY_SITE_COUNT[self.geometry]
        if n_total is None:
            return None
        return n_total - len(self.internal_satisfiers)


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
    # reference restype supplying each torsion potential; None = no reference
    rama_reference: Optional[str] = None
    dunbrack_reference: Optional[str] = None
    na_base_reference: Optional[str] = None
    # D-AAs: residue is mirror image of reference
    reference_mirrored: bool = False
    # One-letter sequence code; unique only within a backbone type ...
    #   "a" is both DA and RA.
    one_letter_code: Optional[str] = None
    # Attachment atom, partner equivalence class, partner atom. Equal atom
    # inventories can need different parameters (e.g. ester vs thioester).
    conjugation_context: Tuple[Tuple[str, str, str], ...] = ()
    # One entry per metal atom this residue carries; empty for everything else.
    metal_sites: Tuple[MetalSite, ...] = ()
    # Output only, never read by scoring: the chemical (Lewis) order of each
    # bond whose `bonds` order is a topology order, as (atom, atom, order).
    io_bond_orders: Tuple[Tuple[str, str, str], ...] = ()

    def atom_name(self, index: int) -> str:
        """Return the name of the atom at ``index``."""
        return self.atoms[index].name


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IcoorVariant:
    """Patch operation that adds or modifies an internal coordinate."""

    name: str
    source: Optional[str] = None
    # absent means "take the source's", so 0.0 can be asked for outright
    phi: Optional[DihedralAngle] = None
    theta: Optional[BondAngle] = None
    d: Optional[float] = None
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
    name3_aliases: Tuple[Name3Alias, ...] = ()

    @classmethod
    def get_default(cls) -> "ChemicalDatabase":
        """Load and return default parameter database."""
        if cls.__default is None:
            cls.__default = ChemicalDatabase.from_file(
                os.path.join(
                    os.environ.get(
                        "TMOL_JAX_DATABASE",
                        os.path.join(os.path.dirname(__file__), "..", "default"),
                    ),
                    "chemical",
                )
            )
        return cls.__default

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ChemicalDatabase":
        """Load chemical definitions and their generated residue tables."""
        with open(os.path.join(path, "chemical.yaml"), "r") as infile:
            raw = safe_load(infile)
        # residue sets written by a support script live in their own files so
        #    regenerating one does not rewrite the hand-maintained database
        for generated in GENERATED_RESIDUE_FILES:
            generated_path = os.path.join(path, generated)
            if not os.path.exists(generated_path):
                continue
            with open(generated_path, "r") as infile:
                raw["residues"].extend(safe_load(infile)["residues"])
        raw = normalize_bond_tuples(raw)

        return _structure(raw, cls)
