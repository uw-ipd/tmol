"""Declarative description of each polymer backbone the NCAA path supports.

A noncanonical polymer residue is prepared as a molecule: its polymer connections
are replaced by chemical stubs (caps), the ligand pipeline types and charges the
capped molecule, then the caps are stripped back to connections. A profile says
which atoms form the backbone, what the caps are, and what the resulting residue
type must declare.
"""

import math
from typing import Optional, Tuple

import attr
import numpy
import biotite.structure as struc


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CapAtom:
    """One stub atom, placed by internal coordinates against three placed atoms."""

    name: str
    element: str
    # frame for place_atom(a, b, c, ...); a residue with only two heavy atoms
    #    cannot supply three, so a two-name frame gets a synthetic first point
    refs: Tuple[str, ...]
    d: float
    angle: float
    dihedral: float
    bond_to: str
    bond_order: str = "SINGLE"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PolymerProfile:
    """How to cap, build and declare one class of polymer residue."""

    name: str
    polymer_type: str
    backbone_type: str
    # geometry/type donor for the backbone transplant
    reference_restype: Optional[str]
    mainchain_atoms: Tuple[str, ...]
    # (connection name, atom it attaches to); None where the residue does not
    #    continue the chain, as a terminal cap does not
    down: Optional[Tuple[str, str]]
    up: Optional[Tuple[str, str]]
    connection_bond_type: str
    caps: Tuple[CapAtom, ...]
    # cap atom standing in for the neighbour across each connection
    down_partner: Optional[str]
    up_partner: Optional[str]
    # mainchain atom whose non-mainchain heavy neighbours root the sidechain;
    #    identified structurally, so no sidechain atom name is ever assumed.
    #    None where the backbone is the whole residue
    sidechain_root_atom: Optional[str]
    # backbone atoms are retyped to protein types; sidechain keeps ligand types
    backbone_types: Tuple[Tuple[str, str], ...]
    # mainchain amide N, (carrying a hydrogen, substituted)
    amide_n_types: Tuple[str, str]
    # (parent mainchain atom, type) for the hydrogens on the backbone
    backbone_h_types: Tuple[Tuple[str, str], ...]
    # named backbone torsions; "conn:n" is the atom n bonds past a connection
    mainchain_torsions: Tuple[Tuple[str, Tuple[str, str, str, str]], ...]
    # icoors taken verbatim from reference_restype; these records are identical
    #    across every canonical residue, so transplanting them keeps the
    #    backbone on database geometry instead of the generated conformer's
    transplant_icoors: Tuple[str, ...]
    # heavy atoms the residue must carry exactly, rather than at least. A cap
    #    is backbone all the way through, so a subset test would claim any
    #    linked fragment carrying the same a couple of atoms
    exact_heavy_atoms: Tuple[str, ...] = ()

    @property
    def cap_names(self):
        return tuple(c.name for c in self.caps)

    @property
    def connections(self):
        """(name, atom) of each polymer connection the backbone actually has."""
        return tuple(c for c in (self.down, self.up) if c is not None)

    @property
    def connection_partners(self):
        """(connection name, cap atom standing in for the neighbour) per connection."""
        return tuple(
            (conn[0], partner)
            for conn, partner in (
                (self.down, self.down_partner),
                (self.up, self.up_partner),
            )
            if conn is not None
        )

    def required_atoms(self):
        """Backbone atoms an input residue must carry to match this profile."""
        return frozenset(self.mainchain_atoms) | {a for _, a in self.connections}


ALPHA_AA = PolymerProfile(
    name="alpha",
    polymer_type="amino_acid",
    backbone_type="alpha",
    reference_restype="ALA",
    mainchain_atoms=("N", "CA", "C"),
    down=("down", "N"),
    up=("up", "C"),
    connection_bond_type="AROMATIC",
    caps=(
        # acetyl on N
        CapAtom("CY", "C", ("C", "CA", "N"), 1.335, 121.0, 180.0, "N"),
        CapAtom("OY", "O", ("CA", "N", "CY"), 1.231, 123.0, 0.0, "CY", "DOUBLE"),
        CapAtom("CAY", "C", ("CA", "N", "CY"), 1.508, 116.0, 180.0, "CY"),
        # n-methylamide on C
        CapAtom("NM", "N", ("N", "CA", "C"), 1.335, 116.2, 180.0, "C"),
        CapAtom("CM", "C", ("CA", "C", "NM"), 1.449, 121.7, 180.0, "NM"),
    ),
    down_partner="CY",
    up_partner="NM",
    sidechain_root_atom="CA",
    backbone_types=(("CA", "CAbb"), ("C", "CObb"), ("O", "OCbb")),
    amide_n_types=("Nbb", "Npro"),
    backbone_h_types=(("N", "HNbb"), ("CA", "Hapo")),
    mainchain_torsions=(
        ("phi", ("down:0", "N", "CA", "C")),
        ("psi", ("N", "CA", "C", "up:0")),
        ("omega", ("CA", "C", "up:0", "up:1")),
    ),
    transplant_icoors=("N", "CA", "C", "up", "O", "down"),
)

# Both caps model as n-methylacetamide: the acetyl cap needs an amide to be a
# molecule, and the methylamide cap needs an acetyl, so each supplies the other
# half. Neither carries a sidechain, and neither continues the chain past
# itself, so each declares one connection.

ACETYL_CAP = PolymerProfile(
    name="acetyl_cap",
    polymer_type="amino_acid",
    backbone_type="alpha",
    reference_restype=None,
    mainchain_atoms=("CH3", "C"),
    down=None,
    up=("up", "C"),
    connection_bond_type="AROMATIC",
    caps=(
        # n-methylamide on C; O and CH3 frame it, the two being coplanar with it
        CapAtom("NM", "N", ("O", "CH3", "C"), 1.335, 116.0, 180.0, "C"),
        CapAtom("CM", "C", ("CH3", "C", "NM"), 1.449, 121.7, 180.0, "NM"),
    ),
    down_partner=None,
    up_partner="NM",
    sidechain_root_atom=None,
    backbone_types=(("CH3", "CH3"), ("C", "CObb"), ("O", "OCbb")),
    amide_n_types=None,
    backbone_h_types=(("CH3", "Hapo"),),
    mainchain_torsions=(),
    transplant_icoors=(),
    exact_heavy_atoms=("CH3", "C", "O"),
)

METHYLAMIDE_CAP = PolymerProfile(
    name="methylamide_cap",
    polymer_type="amino_acid",
    backbone_type="alpha",
    reference_restype=None,
    mainchain_atoms=("N", "C"),
    down=("down", "N"),
    up=None,
    connection_bond_type="AROMATIC",
    caps=(
        # acetyl on N; two heavy atoms cannot frame the first stub, so it takes
        #    a synthetic reference and the rest hang off it
        CapAtom("CY", "C", ("C", "N"), 1.335, 121.0, 180.0, "N"),
        CapAtom("OY", "O", ("C", "N", "CY"), 1.231, 123.0, 0.0, "CY", "DOUBLE"),
        CapAtom("CAY", "C", ("C", "N", "CY"), 1.508, 116.0, 180.0, "CY"),
    ),
    down_partner="CY",
    up_partner=None,
    sidechain_root_atom=None,
    backbone_types=(("N", "Nbb"), ("C", "CH3")),
    amide_n_types=("Nbb", "Nbb"),
    backbone_h_types=(("N", "HNbb"), ("C", "Hapo")),
    mainchain_torsions=(),
    transplant_icoors=(),
    exact_heavy_atoms=("N", "C"),
)

PROFILES = (ALPHA_AA, ACETYL_CAP, METHYLAMIDE_CAP)


def place_atom(a, b, c, d, angle, dihedral):
    """Place a point at distance d from c, angle b-c-x, dihedral a-b-c-x."""
    ang = math.radians(angle)
    dih = math.radians(dihedral)
    bc = c - b
    bc = bc / numpy.linalg.norm(bc)
    n = numpy.cross(b - a, bc)
    n = n / numpy.linalg.norm(n)
    m = numpy.cross(n, bc)
    offset = numpy.array(
        [
            -d * math.cos(ang),
            d * math.sin(ang) * math.cos(dih),
            d * math.sin(ang) * math.sin(dih),
        ]
    )
    return c + offset[0] * bc + offset[1] * m + offset[2] * n


def synthetic_reference(b, c):
    """A third frame point for a residue with too few atoms to supply one.

    Only its direction matters: the caps are scaffolding, stripped again once
    the residue is typed, so any point off the b-c line serves.
    """
    axis = c - b
    axis = axis / numpy.linalg.norm(axis)
    trial = numpy.array([1.0, 0.0, 0.0])
    if abs(float(numpy.dot(trial, axis))) > 0.9:
        trial = numpy.array([0.0, 1.0, 0.0])
    perp = numpy.cross(axis, trial)
    return b + perp / numpy.linalg.norm(perp)


def heavy_atom_names(atom_array):
    return {
        str(n)
        for n, e in zip(atom_array.atom_name, atom_array.element)
        if str(e) != "H"
    }


def profile_for_atom_array(atom_array) -> Optional[PolymerProfile]:
    """The profile whose backbone atoms are all present, or None."""
    names = {str(n) for n in atom_array.atom_name}
    heavy = heavy_atom_names(atom_array)
    for profile in PROFILES:
        if profile.exact_heavy_atoms:
            if heavy == set(profile.exact_heavy_atoms):
                return profile
        elif profile.required_atoms() <= names:
            return profile
    return None


def resolve_cap_names(profile: PolymerProfile, existing):
    """Map each cap atom to a name the residue does not already use.

    Cap names are only scaffolding, but noncanonicals do use names like CM, so
    they cannot be fixed strings.
    """
    taken = set(existing)
    resolved = {}
    for cap in profile.caps:
        name = cap.name
        suffix = 0
        while name in taken:
            suffix += 1
            name = f"{cap.name}{suffix}"
        resolved[cap.name] = name
        taken.add(name)
    return resolved


def cap_residue(atom_array, profile: PolymerProfile):
    """Return (capped heavy-atom AtomArray, cap name mapping).

    Hydrogens are dropped: the pipeline derives a SMILES from heavy atoms and the
    bond table, then re-protonates.
    """
    names = {str(n) for n in atom_array.atom_name}
    missing = sorted(profile.required_atoms() - names)
    if missing:
        raise ValueError(f"residue is missing backbone atom(s) {missing}")
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        raise ValueError(
            "residue carries no bond table; read the structure with "
            "include_bonds=True so its chemistry can be derived"
        )
    cap_names = resolve_cap_names(profile, names)

    keep = numpy.array([str(e) != "H" for e in atom_array.element])
    kept_indices = numpy.nonzero(keep)[0]
    residue = atom_array[kept_indices]

    pos = {str(n): c for n, c in zip(residue.atom_name, residue.coord)}
    for cap in profile.caps:
        frame = [pos[cap_names.get(r, r)] for r in cap.refs]
        if len(frame) == 2:
            frame = [synthetic_reference(*frame)] + frame
        a, b, c = frame
        pos[cap_names[cap.name]] = place_atom(a, b, c, cap.d, cap.angle, cap.dihedral)

    n_residue = residue.array_length()
    out = struc.AtomArray(n_residue + len(profile.caps))
    out.coord[:n_residue] = residue.coord
    out.atom_name[:n_residue] = residue.atom_name
    out.element[:n_residue] = residue.element
    for offset, cap in enumerate(profile.caps):
        out.coord[n_residue + offset] = pos[cap_names[cap.name]]
        out.atom_name[n_residue + offset] = cap_names[cap.name]
        out.element[n_residue + offset] = cap.element
    for field in ("res_name", "chain_id", "res_id", "hetero"):
        if field in atom_array.get_annotation_categories():
            value = getattr(atom_array, field)[0]
            getattr(out, field)[:] = value

    remap = {int(old): new for new, old in enumerate(kept_indices)}
    bonds = struc.BondList(out.array_length())
    if atom_array.bonds is not None:
        for i, j, bond_type in atom_array.bonds.as_array():
            if i in remap and j in remap:
                bonds.add_bond(remap[i], remap[j], bond_type)
    index = {str(n): i for i, n in enumerate(out.atom_name)}
    order = {"SINGLE": struc.BondType.SINGLE, "DOUBLE": struc.BondType.DOUBLE}
    for cap in profile.caps:
        bonds.add_bond(
            index[cap_names[cap.name]],
            index[cap_names.get(cap.bond_to, cap.bond_to)],
            order[cap.bond_order],
        )
    out.bonds = bonds
    return out, cap_names
