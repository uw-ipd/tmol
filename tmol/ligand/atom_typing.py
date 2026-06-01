"""Atom type assignment for ligand atoms.

Assigns Rosetta generic_potential atom types to atoms in an RDKit Mol.
The classification logic is a faithful port of Rosetta's AtomTypeClassifier
(from mol2genparams / generic_potential) and produces identical atom types
and atom names, including the polar-carbon modifier and the Rosetta hydrogen
naming convention (H<bonded_element><count>).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from tmol.ligand.chemistry_tables import get_polar_classes

logger = logging.getLogger(__name__)


# Rosetta hybridization convention (matches mol2genparams)
HYB_SP = 1
HYB_SP2 = 2
HYB_SP3 = 3
HYB_AMIDE = 8
HYB_AROMATIC = 9

ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

# Rosetta Types.CONJUGATING_ACLASSES
_CONJUGATING_ATOM_CLASSES = frozenset(
    {
        "CD",
        "CD1",
        "CD2",
        "CR",
        "CDp",
        "CRp",
        "Nad",
        "Nin",
        "Nim",
        "Ngu1",
        "Ngu2",
        "NG2",
        "NG21",
        "NG22",
        "Nad3",
        "Ofu",
        "OG2",
        "Ssl",
        "SG2",
    }
)

# Map RDKit HybridizationType enum -> legacy OB integer convention
# NOTE: Aromatic is handled separately (returns 9) via IsAromatic
_HYB_MAP = {
    Chem.HybridizationType.S: 3,
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 3,
    Chem.HybridizationType.UNSPECIFIED: 3,
    Chem.HybridizationType.OTHER: 3,
}


class AtomTypeAssignment(NamedTuple):
    atom_name: str
    atom_type: str
    element: str
    index: int


@dataclass
class RosettaTypingState:
    """Precomputed state consumed by Rosetta-style classifiers."""

    source_subtype_by_idx: dict[int, str]
    hyb_by_idx: dict[int, int]
    atms_aro: set[int] = field(default_factory=set)
    atms_strained: set[int] = field(default_factory=set)
    rings: list[tuple[int, ...]] = field(default_factory=list)
    ring_membership_by_idx: dict[int, set[int]] = field(default_factory=dict)
    neighbor_counts_by_idx: dict[int, tuple[int, int, int, int, int, int]] = field(
        default_factory=dict
    )


def _elem_symbol(atomic_num: int) -> str:
    """Return the element symbol for an atomic number.

    Args:
        atomic_num: Atomic number to convert.

    Returns:
        Element symbol string.
    """
    sym = ELEMENT_SYMBOLS.get(atomic_num)
    if sym is not None:
        return sym
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)


def _is_hydrogen(atom: Chem.Atom) -> bool:
    """Return whether an RDKit atom is hydrogen.

    Args:
        atom: Atom to inspect.

    Returns:
        ``True`` if the atom atomic number is 1.
    """
    return atom.GetAtomicNum() == 1


def sanitize_tolerant(mol: Chem.Mol) -> None:
    """Run ``Chem.SanitizeMol`` with a Kekulé/valence-tolerant fallback.

    Some ligand inputs (e.g. guanidinium / amidine groups, fused
    heteroaromatics with mixed Kekulé/aromatic perception, formal-charge
    nitrogens) cannot be kekulized or pass RDKit's strict valence model
    on first try. We retry with sanitization that skips KEKULIZE,
    SETAROMATICITY, and PROPERTIES — preserving the incoming bond
    orders / aromaticity flags rather than dropping the mol on the floor.

    When the molecule carries explicit aromatic flags from the source
    (see :func:`source_carried_kekule`), we skip ``SETAROMATICITY`` /
    ``KEKULIZE`` from the start so RDKit's re-perception cannot
    overwrite the source-supplied flags / Kekulé bond orders.
    """
    from tmol.ligand.rdkit_mol import (
        normalize_non_ring_aromatic_bonds,
        source_has_aromatic_annotations,
        source_carried_kekule,
    )

    if source_carried_kekule(mol) or source_has_aromatic_annotations(mol):
        ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
        try:
            Chem.SanitizeMol(mol, sanitizeOps=ops)
            return
        except (
            Chem.rdchem.KekulizeException,
            Chem.rdchem.AtomKekulizeException,
            Chem.rdchem.AtomValenceException,
        ):
            # Retry after normalizing non-ring aromatic placeholders.
            normalize_non_ring_aromatic_bonds(mol)
            try:
                Chem.SanitizeMol(mol, sanitizeOps=ops)
                return
            except (
                Chem.rdchem.KekulizeException,
                Chem.rdchem.AtomKekulizeException,
                Chem.rdchem.AtomValenceException,
            ):
                pass
    try:
        Chem.SanitizeMol(mol)
    except (
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        Chem.rdchem.AtomValenceException,
    ):
        normalize_non_ring_aromatic_bonds(mol)
        ops = (
            Chem.SANITIZE_ALL
            ^ Chem.SANITIZE_KEKULIZE
            ^ Chem.SANITIZE_SETAROMATICITY
            ^ Chem.SANITIZE_PROPERTIES
        )
        Chem.SanitizeMol(mol, sanitizeOps=ops)


_WAS_AROMATIC_PROP = "_tmol_was_aromatic"
_IS_STRAINED_RING_ATOM_PROP = "_tmol_is_strained_ring_atom"


def _save_aromatic_perception(mol: Chem.Mol) -> None:
    """Stamp each atom's current ``GetIsAromatic`` flag onto an atom prop.

    Kekulization clears the live aromatic flag, but the classifier still
    needs to know that a pyrrole-type N or aromatic ring carbon was once
    aromatic to choose ``Nin`` over ``NG21``. Read back via
    :func:`was_aromatic`.
    """
    for atom in mol.GetAtoms():
        atom.SetIntProp(_WAS_AROMATIC_PROP, 1 if atom.GetIsAromatic() else 0)


def was_aromatic(atom: Chem.Atom) -> bool:
    """Return the aromatic flag captured before kekulization, falling back
    to the live flag if the property wasn't set."""
    if atom.HasProp(_WAS_AROMATIC_PROP):
        return atom.GetIntProp(_WAS_AROMATIC_PROP) == 1
    return atom.GetIsAromatic()


def _annotate_strained_ring_atoms(mol: Chem.Mol) -> None:
    """Mark atoms in 3/4-membered rings for Rosetta-aligned CSQ typing."""
    for atom in mol.GetAtoms():
        atom.SetIntProp(_IS_STRAINED_RING_ATOM_PROP, 0)
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) <= 4:
            for idx in ring:
                mol.GetAtomWithIdx(idx).SetIntProp(_IS_STRAINED_RING_ATOM_PROP, 1)


def _is_strained_ring_atom(atom: Chem.Atom) -> bool:
    """Return whether an atom was annotated as a strained-ring member.

    Args:
        atom: Atom to inspect.

    Returns:
        ``True`` when the atom belongs to a 3/4-membered ring.
    """
    return atom.HasProp(_IS_STRAINED_RING_ATOM_PROP) and (
        atom.GetIntProp(_IS_STRAINED_RING_ATOM_PROP) == 1
    )


def _is_amide_like_nitrogen(atom: Chem.Atom) -> bool:
    """True if a nitrogen is bonded to a carbonyl carbon (amide / carbamate / urea)."""
    if atom.GetAtomicNum() != 7:
        return False

    for bond in atom.GetBonds():
        c = bond.GetOtherAtom(atom)
        if c.GetAtomicNum() != 6:
            continue
        if any(
            cb.GetBondTypeAsDouble() == 2.0 and cb.GetOtherAtom(c).GetAtomicNum() == 8
            for cb in c.GetBonds()
        ):
            return True

    return False


def _hyb_from_atom_and_subtype(atom: Chem.Atom, subtype: str) -> int:
    """Map subtype + atom perception to Rosetta-style hybridization code."""
    s = subtype.lower()

    # Trust RDkit for aromaticity perception (if in a ring)
    if was_aromatic(atom) and atom.IsInRing():
        return HYB_AROMATIC

    # Handle amide perceptions (as those bonds are often kekulization-sensitive)
    if _is_amide_like_nitrogen(atom):
        return HYB_AMIDE

    # Rosetta Types.SPECIAL_HYBRIDS + observed Tripos/GAFF suffixes.
    subtype_hyb_map = {
        # Tripos/mol2
        "ar": HYB_AROMATIC,
        "aro": HYB_SP2,
        "1": HYB_SP,
        "2": HYB_SP2,
        "3": HYB_SP3,
        "4": HYB_SP3,
        "am": HYB_AMIDE,
        "pl3": HYB_SP2,
        "cat": HYB_SP2,
        "co2": HYB_SP2,
        # Sulfur/P variants
        "o": HYB_SP2,
        "o2": 5,
        "s.o2": 5,
        # GAFF-like variants from Rosetta SPECIAL_HYBRIDS
        "c": HYB_SP2,
        "ca": HYB_AROMATIC,
        "c1": HYB_SP,
        "c2": HYB_SP2,
        "c3": HYB_SP3,
        "cc": HYB_SP2,
        "cd": HYB_SP2,
        "ce": HYB_SP2,
        "cf": HYB_SP2,
        "ch": HYB_SP,
        "cg": HYB_SP,
        "cp": HYB_SP2,
        "cx": HYB_SP2,
        "cy": HYB_SP3,
        "cz": HYB_SP2,
        "n": HYB_SP2,
        "na": HYB_AROMATIC,
        "nb": HYB_SP2,
        "nc": HYB_SP2,
        "nd": HYB_SP2,
        "ne": HYB_SP2,
        "nf": HYB_SP2,
        "no": HYB_SP2,
        "n1": HYB_SP,
        "n2": HYB_SP2,
        "n3": HYB_SP3,
        "n4": HYB_SP3,
        "n.4": HYB_SP3,
        # Rosetta keeps nh as 0 and fills later.
        "nh": 0,
        "os": HYB_SP3,
        "oh": HYB_SP3,
        "s": HYB_SP3,
        "ss": HYB_SP3,
        "sh": HYB_SP3,
        "s4": HYB_SP2,
        "s6": 5,
        "sy": 5,
    }
    if s in subtype_hyb_map:
        return subtype_hyb_map[s]
    if atom.GetIsAromatic():
        return HYB_AROMATIC
    return _HYB_MAP.get(atom.GetHybridization(), HYB_SP3)


def _is_ring_planar(
    mol: Chem.Mol, ring: tuple[int, ...], threshold: float = 1.0e-2
) -> bool:
    """Rosetta-like planarity gate for long aromatic rings."""
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    coords = []
    for idx in ring:
        p = conf.GetAtomPosition(idx)
        coords.append((float(p.x), float(p.y), float(p.z)))
    n = len(coords)
    if n < 3:
        return False
    xyz = np.asarray(coords, dtype=np.float64)
    xyz -= np.mean(xyz, axis=0)
    cov = np.cov(xyz, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    min_eval = float(np.min(evals))
    return min_eval <= threshold


def _assign_missing_hybridization(
    mol: Chem.Mol, hyb_by_idx: dict[int, int], atms_aro: set[int]
) -> None:
    """Port Rosetta assign_hybridization_if_missing for hyb==0 cases."""
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    def _dihedral_deg(i: int, j: int, k: int, l: int) -> float:
        """Compute dihedral angle for four atom indices.

        Args:
            i: First atom index.
            j: Second atom index.
            k: Third atom index.
            l: Fourth atom index.

        Returns:
            Signed dihedral angle in degrees.
        """
        if conf is None:
            return 180.0
        p1 = conf.GetAtomPosition(i)
        p2 = conf.GetAtomPosition(j)
        p3 = conf.GetAtomPosition(k)
        p4 = conf.GetAtomPosition(l)

        b0 = (float(p1.x - p2.x), float(p1.y - p2.y), float(p1.z - p2.z))
        b1 = (float(p3.x - p2.x), float(p3.y - p2.y), float(p3.z - p2.z))
        b2 = (float(p4.x - p3.x), float(p4.y - p3.y), float(p4.z - p3.z))

        def dot(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
            """Return dot product of two 3D vectors."""
            return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

        def cross(
            u: tuple[float, float, float], v: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            """Return cross product of two 3D vectors."""
            return (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )

        def norm(u: tuple[float, float, float]) -> float:
            """Return Euclidean norm of a 3D vector."""
            return math.sqrt(dot(u, u))

        b1n = norm(b1)
        if b1n < 1.0e-8:
            return 180.0
        b1u = (b1[0] / b1n, b1[1] / b1n, b1[2] / b1n)
        v = cross(b0, b1u)
        w = cross(b2, b1u)
        nv = norm(v)
        nw = norm(w)
        if nv < 1.0e-8 or nw < 1.0e-8:
            return 180.0
        v = (v[0] / nv, v[1] / nv, v[2] / nv)
        w = (w[0] / nw, w[1] / nw, w[2] / nw)
        x = max(-1.0, min(1.0, dot(v, w)))
        y = dot(cross(v, w), b1u)
        return math.degrees(math.atan2(y, x))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if hyb_by_idx.get(idx) != 0:
            continue
        attached_to_aro = False
        attached_to_nonaro = False
        n_h_attached = 0
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j in atms_aro:
                attached_to_aro = True
            elif nbr.GetAtomicNum() == 1:
                n_h_attached += 1
            else:
                attached_to_nonaro = True
        # Rosetta special case for N attached to aromatic atoms.
        if atom.GetAtomicNum() == 7 and attached_to_aro:
            if n_h_attached == 2 or not attached_to_nonaro:
                hyb_by_idx[idx] = HYB_SP2
            else:
                hyb_by_idx[idx] = HYB_SP3
            continue
        if atom.GetDegree() < 3:
            hyb_by_idx[idx] = HYB_SP2
            continue

        # Rosetta assign_hybridization_if_missing uses improper torsion:
        # near-planar (<10 degrees) -> sp2, else sp3.
        neigh = [n.GetIdx() for n in atom.GetNeighbors()][:3]
        if len(neigh) < 3:
            hyb_by_idx[idx] = HYB_SP2
            continue
        imp = abs(_dihedral_deg(neigh[0], neigh[1], neigh[2], idx))
        hyb_by_idx[idx] = HYB_SP2 if imp < 10.0 else HYB_SP3


def _is_rosetta_aromatic_ring(
    mol: Chem.Mol, ring: tuple[int, ...], hyb_by_idx: dict[int, int]
) -> bool:
    """Mirror Rosetta `classify_ring_type` aromatic eligibility."""
    ring_size = len(ring)
    if ring_size <= 4:
        return False
    if ring_size > 6:
        return _is_ring_planar(mol, ring)

    is_aro = True
    nsp2_n = 0
    nsp3 = 0
    nsp3_os = 0
    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        hyb = hyb_by_idx[idx]
        z = atom.GetAtomicNum()
        if hyb == HYB_SP3:
            nsp3 += 1
            for bond in atom.GetBonds():
                order = bond.GetBondTypeAsDouble()
                if bond.GetIsAromatic() or order in (2.0, 4.0):
                    continue
                is_aro = False
                break
            if z in (8, 16):
                nsp3_os += 1
        elif hyb == HYB_SP2 and z == 7:
            nsp2_n += 1

        if ring_size == 5:
            # Rosetta's 5-membered aromatic exception:
            # one sp3 O/S ring atom or two sp2 nitrogens.
            is_aro = (nsp3 == nsp3_os) or (nsp2_n == 2)

    # Rosetta's 6-member "aromatic-like" rings are effectively all-sp2.
    # Keeping this explicit avoids over-assigning aromaticity when RDKit
    # carries aromatic bond flags on mixed sp2/sp3 fused systems.
    if ring_size == 6 and nsp3 > 0:
        return False

    return is_aro


def _build_rosetta_typing_state(mol: Chem.Mol) -> RosettaTypingState:
    """Construct Rosetta-like typing state consumed by all classifiers."""
    from tmol.ligand.rdkit_mol import source_subtype

    source_subtype_by_idx: dict[int, str] = {}
    hyb_by_idx: dict[int, int] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        subtype = source_subtype(atom)
        source_subtype_by_idx[idx] = subtype
        hyb_by_idx[idx] = _hyb_from_atom_and_subtype(atom, subtype)

    # Rosetta's ring construction is conservative (cycle-edge based),
    # closer to RDKit's canonical SSSR than SymmSSSR's expanded set.
    # Using AtomRings here avoids over-marking aromatic ring membership
    # on bridged/fused systems, which directly affects Nin/Ofu typing.
    rings = [tuple(int(i) for i in ring) for ring in mol.GetRingInfo().AtomRings()]

    ring_membership_by_idx: dict[int, set[int]] = {}
    atms_strained: set[int] = set()
    atms_aro: set[int] = set()
    for ring_id, ring in enumerate(rings):
        ring_size = len(ring)
        for idx in ring:
            ring_membership_by_idx.setdefault(idx, set()).add(ring_id)
        if ring_size <= 4:
            atms_strained.update(ring)
        if _is_rosetta_aromatic_ring(mol, ring, hyb_by_idx):
            atms_aro.update(ring)

    # Fill missing hybs (e.g. nh) after initial ring aromatic assignment.
    _assign_missing_hybridization(mol, hyb_by_idx, atms_aro)

    # Recompute aromatic atom-set with finalized hyb codes.
    atms_aro = set()
    for ring in rings:
        if _is_rosetta_aromatic_ring(mol, ring, hyb_by_idx):
            atms_aro.update(ring)

    neighbor_counts_by_idx: dict[int, tuple[int, int, int, int, int, int]] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        nC = nH = nO = nN = nS = 0
        ntot = 0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            ntot += 1
            if z == 1:
                nH += 1
            elif z == 6:
                nC += 1
            elif z == 7:
                nN += 1
            elif z == 8:
                nO += 1
            elif z == 16:
                nS += 1
        neighbor_counts_by_idx[idx] = (nC, nH, nO, nN, nS, ntot)

    return RosettaTypingState(
        source_subtype_by_idx=source_subtype_by_idx,
        hyb_by_idx=hyb_by_idx,
        atms_aro=atms_aro,
        atms_strained=atms_strained,
        rings=rings,
        ring_membership_by_idx=ring_membership_by_idx,
        neighbor_counts_by_idx=neighbor_counts_by_idx,
    )


def _state_for_atom(
    atom: Chem.Atom, state: RosettaTypingState | None
) -> RosettaTypingState:
    """Return typing state, lazily constructing one from atom ownership.

    Args:
        atom: Atom that anchors the owning molecule.
        state: Optional precomputed typing state.

    Returns:
        A valid ``RosettaTypingState`` instance.
    """
    if state is not None:
        return state
    return _build_rosetta_typing_state(atom.GetOwningMol())


def kekulize_tolerant(mol: Chem.Mol) -> None:
    """Force Kekulé bond orders + clear aromatic flags, tolerant of failure.

    Rosetta's atom-type classifier and the reference ``.tmol`` files use
    Kekulé conventions (sp2 ``CD/CD1/CDp`` rather than aromatic
    ``CR/CRp``; explicit ``DOUBLE``/``SINGLE`` ring bonds rather than
    ``AROMATIC``). Aromaticity perception in RDKit's standard sanitize
    flips them back, so we kekulize after every sanitize step.

    Stamps :func:`was_aromatic` first so downstream classifiers can still
    distinguish pyrrole-type N (``Nin``) from plain sp2 NH (``NG21``).
    """
    _save_aromatic_perception(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomKekulizeException):
        pass


def should_kekulize_for_typing(mol: Chem.Mol) -> bool:
    """``True`` when this mol's rings should use Kekulé typing.

    Rosetta's mol2genparams takes atom types directly from the source
    mol2's atom-type column. mol2 files written with sp2 (``C.2``) want
    Kekulé / ``CD`` typing; SMILES sources and mol2 files with ``C.ar``
    want aromatic / ``CR``. We can't recover the column itself, but we
    can read a flag set when the source AtomArray carried explicit
    Kekulé bond orders (see ``rdkit_mol.source_carried_kekule``).
    """
    from tmol.ligand.rdkit_mol import source_carried_kekule

    return source_carried_kekule(mol)


def _prepare_mol_for_typing(mol: Chem.Mol) -> Chem.Mol:
    """Normalize RDKit perception so it matches Rosetta expectations.

    Idempotent when the mol has already been sanitized + hydrogenated by
    the main pipeline; required for direct unit-test entry points.
    """
    if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)

    sanitize_tolerant(mol)

    if should_kekulize_for_typing(mol):
        # Rosetta's atom-type pipeline for non-.ar sources expects explicit
        # single/double bond orders during classification.
        kekulize_tolerant(mol)

    # Perceive rings
    Chem.GetSSSR(mol)

    if mol.GetNumConformers() > 0:
        # We have conformers - assign stereochemistry from 3D geometry
        Chem.AssignStereochemistryFrom3D(mol)
    else:
        # No conformers - assign stereochemistry from available info (e.g. chiral tags, E/Z bonds)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    return mol


def _ensure_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit hydrogens when a Mol has none.

    Rosetta's degree-based carbon typing assumes explicit H neighbors are
    present; without them, saturated carbons can be misclassified as CD/CT.
    Returns the (possibly new) mol so the caller can rebind.
    """
    if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        # Only add coordinates if we have a conformer
        mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
    return mol


def _neighbor_counts(
    atom: Chem.Atom, state: RosettaTypingState | None = None
) -> tuple[int, int, int, int, int, int]:
    """Return (nC, nH, nO, nN, nS, ntot) for neighbors of atom."""
    if state is not None:
        idx = atom.GetIdx()
        if idx in state.neighbor_counts_by_idx:
            return state.neighbor_counts_by_idx[idx]

    nC = nH = nO = nN = nS = 0
    ntot = 0
    for nbr in atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        ntot += 1
        if z == 1:
            nH += 1
        elif z == 6:
            nC += 1
        elif z == 7:
            nN += 1
        elif z == 8:
            nO += 1
        elif z == 16:
            nS += 1
    return nC, nH, nO, nN, nS, ntot


def _get_hyb(atom: Chem.Atom, state: RosettaTypingState | None = None) -> int:
    """Map RDKit hybridization to Rosetta mol2 convention.

    Returns: 1=sp, 2=sp2, 3=sp3, 9=aromatic.
    Aromatic is checked first via GetIsAromatic(); other hybridizations
    map through _HYB_MAP with sp3 (3) as the default for unknown types.
    """
    if state is not None:
        idx = atom.GetIdx()
        if idx in state.hyb_by_idx:
            return state.hyb_by_idx[idx]

    from tmol.ligand.rdkit_mol import source_subtype

    return _hyb_from_atom_and_subtype(atom, source_subtype(atom))


def _has_sp2_oxygen_neighbor(
    atom: Chem.Atom, state: RosettaTypingState | None = None
) -> bool:
    """Rosetta amide proxy: carbon attached to any O(sp2) neighbor."""
    for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)
        if nbr.GetAtomicNum() == 8 and _get_hyb(nbr, state) == 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-element classifiers — ported 1:1 from Rosetta AtomTypeClassifier
# ---------------------------------------------------------------------------


def _classify_H(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify hydrogen atom type from its bonded heavy atom.

    Args:
        atom: Hydrogen atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Rosetta generic-potential atom type for hydrogen.
    """
    state = _state_for_atom(atom, state)
    for nbr in atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z == 6:
            return "HR" if _get_hyb(nbr, state) == 9 else "HC"
        elif z == 8:
            return "HO"
        elif z == 7:
            return "HN"
        elif z == 16:
            return "HS"
        else:
            return "HG"
    return "HG"


def _classify_C(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify carbon atom type using Rosetta-like rules.

    Args:
        atom: Carbon atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Rosetta generic-potential carbon type.
    """
    state = _state_for_atom(atom, state)
    hyb = _get_hyb(atom, state)
    nbonds = atom.GetDegree()
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom, state)

    sub = state.source_subtype_by_idx.get(atom.GetIdx(), "")
    if sub == "ar" or hyb == HYB_AROMATIC:
        prefix = "CR"
    elif sub:
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            prefix = "CS"
        elif nbonds == 4:
            prefix = "CS"
        elif nbonds == 3:
            prefix = "CD"
        elif nbonds <= 2:
            prefix = "CT"
        else:
            return "CS"
    elif hyb == 9:
        prefix = "CR"
    elif atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        prefix = "CS"
    elif nbonds == 4:
        prefix = "CS"
    elif nbonds == 3:
        prefix = "CD"
    elif nbonds <= 2:
        prefix = "CT"
    else:
        return "CS"

    if prefix == "CS" and atom.GetIdx() in state.atms_strained:
        # Rosetta classify_C() upgrades sp3 carbons in 3/4-membered rings to CSQ.
        return "CSQ"
    if prefix in ("CS", "CD"):
        if nH > 0:
            return f"{prefix}{nH}"
        return prefix
    return prefix


def _classify_N_hetero(
    atom: Chem.Atom,
    hyb: int,
    nC: int,
    nH: int,
    nO: int,
    nN: int,
    ntot: int,
    state: RosettaTypingState | None = None,
) -> str:
    """Classify nitrogen with non-C/H neighbors (lone pairs, heteroatoms)."""
    state = _state_for_atom(atom, state)
    sub = state.source_subtype_by_idx.get(atom.GetIdx(), "").lower()

    def _in_5_or_6_ring() -> bool:
        """Return whether atom belongs to any 5/6-membered ring."""
        ring_ids = state.ring_membership_by_idx.get(atom.GetIdx(), set())
        for rid in ring_ids:
            if rid < len(state.rings):
                n = len(state.rings[rid])
                if n in (5, 6):
                    return True
        return False

    if hyb == 3:
        if ntot <= 3 and nH >= 1:
            return "Nam2"
        elif nH >= 1:
            return "Nam"
        return "NG3"

    if hyb in (2, 8, 9):
        # DUD/reference parity: ring-conjugated tertiary N.pl3/N.am sites
        # with one+ N neighbor and no hydrogens are Nim-like.
        if (
            sub in {"pl3", "am"}
            and nH == 0
            and nN >= 1
            and nC <= 2
            and _in_5_or_6_ring()
        ):
            return "Nim"
        # Reference parity: aromatic 5/6-ring tertiary N(=C)-N motifs with
        # subtype ``N.2`` are Nim-like rather than Nad3.
        if (
            sub in {"2", "ar"}
            and nH == 0
            and nN >= 1
            and nC == 2
            and atom.GetIdx() in state.atms_aro
            and _in_5_or_6_ring()
        ):
            return "Nim"
        if nO >= 2:
            return "NG2"
        if nH == 0 and nN >= 1:
            # Rosetta AtomTypeClassifier assigns Nad3 for hetero-connected
            # sp2/aromatic N with no H and at least one N neighbor.
            return "Nad3"
        if nH == 0:
            return "NG2"
        if nH == 1:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6 and _get_hyb(nbr, state) == 2:
                    if _has_sp2_oxygen_neighbor(nbr, state):
                        return "Nad"
            return "NG21"
        return "NG22"

    return "NG2"


def _n_sp2_context_flags(
    atom: Chem.Atom, state: RosettaTypingState | None = None
) -> tuple[bool, bool, int]:
    """Return (is_amide, is_guanidinium, nCaro_like) around an sp2 N."""
    state = _state_for_atom(atom, state)
    is_guanidinium = False
    is_amide = False
    nCaro = 0

    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        nbr_hyb = _get_hyb(nbr, state)
        if nbr_hyb in (2, 9):
            nCaro += 1
            nOsp2_j = 0
            nNsp2_j = 0
            for nbr_k in nbr.GetNeighbors():
                if nbr_k.GetIdx() == atom.GetIdx():
                    continue
                if nbr_k.GetAtomicNum() == 8 and _get_hyb(nbr_k, state) == 2:
                    nOsp2_j += 1
                elif nbr_k.GetAtomicNum() == 7 and _get_hyb(nbr_k, state) in (2, 8, 9):
                    nNsp2_j += 1
            if nOsp2_j == 1:
                is_amide = True
            if nNsp2_j == 2:
                is_guanidinium = True

    return is_amide, is_guanidinium, nCaro


def _classify_N_sp2(
    atom: Chem.Atom, nC: int, nH: int, state: RosettaTypingState | None = None
) -> str:
    """Classify sp2 nitrogen connected only to C/H."""
    state = _state_for_atom(atom, state)
    is_amide, is_guanidinium, nCaro = _n_sp2_context_flags(atom, state)

    if is_amide:
        if nC == 3:
            return "Nad3"
        return "Nad" if nH >= 1 else "NG2"

    if is_guanidinium:
        if nH == 2:
            return "Ngu2"
        return "Ngu1" if nH == 1 else "NG2"

    if nC == 2 and nCaro >= 1:
        # Pyrrole-type N-H sits in a Hückel-aromatic 5-ring. We must
        # use Rosetta-like aromatic ring membership from typing state.
        is_aro_ring_n = atom.GetIdx() in state.atms_aro
        if nH == 1 and is_aro_ring_n:
            return "Nin"
        if nH == 1:
            return "NG21"
        if nH == 0:
            return "Nim"
        return "NG22"

    if nC == 3 and nCaro >= 1:
        return "Nad3"

    if nH == 0:
        return "NG2"
    return "NG21" if nH == 1 else "NG22"


def _classify_N(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify nitrogen atom type using Rosetta-like rules.

    Args:
        atom: Nitrogen atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Rosetta generic-potential nitrogen type.
    """
    state = _state_for_atom(atom, state)
    hyb = _get_hyb(atom, state)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom, state)

    if hyb == 1:
        return "NG1"

    # Rosetta infer_atomtypes anomaly path:
    # hyb==2, ntot==4, nH==3 => Nam; otherwise protonated sp2 N => Nam2.
    if hyb == 2 and ntot == 4 and nH == 3:
        return "Nam"
    if hyb == 2 and ntot == 4 and 1 <= nH < 3:
        return "Nam2"
    # Protonated amide tertiary N follows the same Nam2 assignment in Rosetta.
    if hyb == 8 and nC == 3 and nH == 1:
        return "Nam2"

    if nC + nH < ntot:
        return _classify_N_hetero(atom, hyb, nC, nH, nO, nN, ntot, state)

    if hyb == 8:
        if nC == 3:
            return "Nad3"
        return "Nad3" if nH == 0 else "Nad"

    if hyb == 2:
        return _classify_N_sp2(atom, nC, nH, state)

    if hyb == 9:
        if ntot == 3 and nC == 2 and nH == 1:
            return "Nin"
        if ntot == 3 and nC == 3:
            return "Nad3"
        if ntot == 2 and nC == 2:
            return "Nim"
        if nH == 0:
            return "NG2"
        return "NG21" if nH == 1 else "NG22"

    if hyb == 3:
        if nH == 0:
            return "NG3"
        return "Nam2" if ntot <= 3 else "Nam"

    return "NG2"


def _classify_O_no_carbon(
    atom: Chem.Atom,
    hyb: int,
    nH: int,
    nN: int,
    ntot: int,
    state: RosettaTypingState | None = None,
) -> str:
    """Classify oxygen not bonded to any carbon."""
    state = _state_for_atom(atom, state)
    if hyb == 3:
        if nH >= 1:
            is_PO4H = False
            if ntot == 2:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 15:
                        is_PO4H = True
                        break
            return "Ohx" if is_PO4H else "OG31"
        return "OG3"
    if hyb == 2:
        # Rosetta infer_atomtypes oxime rescue: O(sp2)-N(sp2)-H.
        if nH >= 1 and nN >= 1:
            n_sp2_n = 0
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 7 and _get_hyb(nbr, state) == 2:
                    n_sp2_n += 1
            if n_sp2_n == 1 and ntot == 2:
                return "OG31"
        return "Ont" if nN == 1 and ntot == 1 else "OG2"
    return "OG2"


def _classify_O_sp2(
    atom: Chem.Atom, nC: int, state: RosettaTypingState | None = None
) -> str:
    """Classify sp2 oxygen bonded to at least one carbon."""
    state = _state_for_atom(atom, state)
    if nC == 2:
        # Rosetta classify_O: sp2 oxygen attached to two carbons => Ofu.
        return "Ofu"

    bonds = list(atom.GetBonds())
    if not bonds:
        return "OG2"
    first_nbr = bonds[0].GetOtherAtom(atom)
    if first_nbr.GetAtomicNum() != 6:
        return "OG2"
    c_nbr = first_nbr

    nC_j, nH_j, nO_j, nN_j, _, _ = _neighbor_counts(c_nbr, state)
    nO_j -= 1

    if nN_j >= 1:
        return "Oad"
    if nC_j == 2 or nC_j + nH_j == 2:
        return "Oal"

    if nO_j == 1:
        for nbr in c_nbr.GetNeighbors():
            if nbr.GetIdx() == atom.GetIdx():
                continue
            if nbr.GetAtomicNum() == 8:
                deg = nbr.GetDegree()
                if deg == 2:
                    return "Oal"
                if deg == 1:
                    return "Oat"
                return "OG2"

    return "OG2"


def _classify_O(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify oxygen atom type using Rosetta-like rules.

    Args:
        atom: Oxygen atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Rosetta generic-potential oxygen type.
    """
    state = _state_for_atom(atom, state)
    hyb = _get_hyb(atom, state)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom, state)
    sub = state.source_subtype_by_idx.get(atom.GetIdx(), "")

    if nC == 0:
        return _classify_O_no_carbon(atom, hyb, nH, nN, ntot, state)

    if ntot > 2:
        return "OG2"

    # Source-mol2 ``O.2`` forces the sp2 branch even when the neighbour
    # carbon is aromatic. Rosetta's mol2genparams reads CR/CD and
    # Oal/Ohx straight from the source column; our phenolic-OH
    # short-circuit is a heuristic that only applies when the source
    # didn't already say sp2.
    if sub != "2" and ntot == 2 and nH >= 1 and nC == 1:
        c_nbr = next(n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_nbr.GetIdx() in state.atms_aro:
            return "Ohx"

    # When source says ``O.2``, route to the sp2 classifier directly.
    if sub == "2":
        # Rosetta infer_atomtypes special-case: oxime-like O (N(sp2)-O-H)
        # can map to OG31 despite source O.2.
        if nH >= 1 and nN >= 1:
            n_sp2_n = 0
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 7 and _get_hyb(nbr, state) == 2:
                    n_sp2_n += 1
            if n_sp2_n == 1 and ntot == 2:
                return "OG31"
        return _classify_O_sp2(atom, nC, state)
    # Source ``O.3`` should follow Rosetta's sp3 oxygen branch even when
    # RDKit drifts toward sp2 due nearby conjugation.
    if sub == "3":
        if nH >= 1:
            return "Ohx"
        if ntot == 2 and nC == 2:
            return "Ofu" if atom.GetIdx() in state.atms_aro else "Oet"
        return "OG31" if nH >= 1 else "OG3"

    if hyb == 3:
        if nH >= 1:
            return "Ohx"
        if ntot == 2 and nC == 2:
            # Rosetta classify_O: sp3 oxygen attached to two carbons is
            # Ofu when aromatic, Oet otherwise.
            return "Ofu" if atom.GetIdx() in state.atms_aro else "Oet"
        return "OG3"

    if hyb == 2 or hyb == 9:
        # Treat aromatic-perceived Os (hyb=9) the same as sp2 — RDKit
        # often marks carboxylate / nitro / heterocyclic oxygens
        # aromatic when their lone pair conjugates into a ring, but the
        # Rosetta classifier wants the Oat / Oal / Oad distinction
        # driven by the bonded carbon's substituent pattern.
        return _classify_O_sp2(atom, nC, state)

    return "OG2"


def _classify_S(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify sulfur atom type using Rosetta-like rules.

    Args:
        atom: Sulfur atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Rosetta generic-potential sulfur type.
    """
    state = _state_for_atom(atom, state)
    nC, nH, _, _, nS, ntot = _neighbor_counts(atom, state)
    if nC == 1 and nH == 1 and ntot == 2:
        return "Sth"
    elif nC + nS == 2 and ntot == 2:
        return "SR" if atom.GetIdx() in state.atms_aro else "Ssl"
    elif ntot == 1:
        return "SG2"
    else:
        return "SG5" if _get_hyb(atom, state) == 5 else "SG3"


def _classify_P(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify phosphorus atom type from Rosetta hybridization code.

    Args:
        atom: Phosphorus atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        ``PG5`` when hypervalent, otherwise ``PG3``.
    """
    state = _state_for_atom(atom, state)
    # Rosetta classify_P is keyed off hybridization: hyb==5 => PG5 else PG3.
    if _get_hyb(atom, state) == 5:
        return "PG5"
    return "PG3"


def _classify_halogen(
    atom: Chem.Atom, mol: Chem.Mol, state: RosettaTypingState | None = None
) -> str:
    """Classify halogen atom type.

    Args:
        atom: Halogen atom to classify.
        mol: Molecule containing ``atom``.
        state: Optional precomputed typing state.

    Returns:
        Halogen class label (``F``, ``Cl``, ``Br``, or ``I``).
    """
    state = _state_for_atom(atom, state)
    z = atom.GetAtomicNum()
    base = {9: "F", 17: "Cl", 35: "Br", 53: "I"}[z]
    if atom.GetDegree() == 1:
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 6 and nbr.GetIdx() in state.atms_aro:
                return base + "R"
    return base


# ---------------------------------------------------------------------------
# modify_polarC — post-classification pass matching Rosetta exactly
# ---------------------------------------------------------------------------


def _modify_polar_c(
    assignments: list[AtomTypeAssignment],
    mol: Chem.Mol,
) -> list[AtomTypeAssignment]:
    """Convert CS/CD/CR/CT -> CSp/CDp/CRp/CTp for polar-adjacent carbons."""
    type_by_idx = {a.index: a.atom_type for a in assignments}
    polar_classes = get_polar_classes()

    result = []
    for a in assignments:
        atom = mol.GetAtomWithIdx(a.index)
        if atom.GetAtomicNum() != 6:
            result.append(a)
            continue

        atype = a.atom_type
        prefix = atype[:2]
        if prefix not in ("CS", "CD", "CR", "CT"):
            result.append(a)
            continue

        n_heavy = sum(1 for nbr in atom.GetNeighbors() if not _is_hydrogen(nbr))
        if n_heavy <= 1:
            result.append(a)
            continue

        attached_to_polar = False
        for nbr in atom.GetNeighbors():
            nbr_type = type_by_idx.get(nbr.GetIdx())
            if nbr_type in polar_classes:
                attached_to_polar = True
                break

        if attached_to_polar:
            new_type = prefix + "p"
            result.append(a._replace(atom_type=new_type))
        else:
            result.append(a)

    return result


# ---------------------------------------------------------------------------
# Ring-based nitrogen correction — matching Rosetta post-classification
# ---------------------------------------------------------------------------


def _correct_ring_nitrogen(
    assignments: list[AtomTypeAssignment],
    mol: Chem.Mol,
    state: RosettaTypingState | None = None,
) -> list[AtomTypeAssignment]:
    """Override nitrogen type to Nim for sp2/aromatic N in 5/6-membered rings.

    Rosetta post-pass semantics:
    in 5/6-membered rings, N atoms with hyb in {2,8,9} and
    (ntot<3, nH==0, nN>=1) are forced to Nim.
    """
    if state is None:
        state = _build_rosetta_typing_state(mol)

    result = list(assignments)

    # Use RDKit's ring membership for the post-pass trigger. The broader
    # SymmSSSR-derived ring set can over-trigger Nad3->Nim corrections in
    # fused systems compared to the historical Rosetta-parity baseline.
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        ring_size = len(ring)
        if ring_size < 5 or ring_size > 6:
            continue

        for i, a in enumerate(result):
            if a.index not in ring:
                continue
            atom = mol.GetAtomWithIdx(a.index)
            if atom.GetAtomicNum() != 7:
                continue
            hyb = _get_hyb(atom, state)
            if hyb not in (HYB_SP2, HYB_AMIDE, HYB_AROMATIC):
                continue

            nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom, state)
            if ntot < 3 and nH == 0 and nN >= 1:
                result[i] = a._replace(atom_type="Nim")

    return result


def _correct_amide_bond_orders(
    assignments: list[AtomTypeAssignment],
    mol: Chem.Mol,
) -> None:
    """Rosetta-style amide bond correction.

    When a Nad/Nad3 nitrogen is single-bonded to CDp carbon, promote the
    bond order to double-like to match Rosetta's post-classification
    amide-bond correction semantics.
    """
    type_by_idx = {a.index: a.atom_type for a in assignments}

    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ti = type_by_idx.get(i, "")
        tj = type_by_idx.get(j, "")
        is_amide = (ti in {"Nad", "Nad3"} and tj == "CDp") or (
            tj in {"Nad", "Nad3"} and ti == "CDp"
        )
        if is_amide:
            bond.SetBondType(Chem.BondType.DOUBLE)
            bond.SetIsAromatic(False)


def _bond_is_planar(mol: Chem.Mol, i: int, j: int, cutoff_deg: float = 40.0) -> bool:
    """Rosetta-like planarity check around a bond.

    Mirrors SetupTopology.is_planar usage as a secondary gate for promoting
    conjugated single bonds.
    """
    if mol.GetNumConformers() == 0:
        return True
    conf = mol.GetConformer()
    ai = mol.GetAtomWithIdx(i)
    aj = mol.GetAtomWithIdx(j)
    ni = [
        n.GetIdx()
        for n in ai.GetNeighbors()
        if n.GetIdx() != j and n.GetAtomicNum() != 1
    ]
    nj = [
        n.GetIdx()
        for n in aj.GetNeighbors()
        if n.GetIdx() != i and n.GetAtomicNum() != 1
    ]
    if not ni or not nj:
        return True

    def _is_near_planar(phi: float) -> bool:
        """Return whether a dihedral angle is within planar cutoff."""
        a = abs(phi)
        delta = min(abs(a), abs(180.0 - a))
        return delta <= cutoff_deg

    for a in ni:
        for b in nj:
            try:
                phi = rdMolTransforms.GetDihedralDeg(
                    conf, int(a), int(i), int(j), int(b)
                )
            except Exception:
                continue
            if _is_near_planar(phi):
                return True
    return False


def _correct_conjugated_single_bond_orders(  # noqa: C901
    assignments: list[AtomTypeAssignment],
    mol: Chem.Mol,
    state: RosettaTypingState,
) -> None:
    """Promote Rosetta-conjugated single bonds to double for params output.

    Equivalent to Rosetta BondClass.order_in_params behavior after
    assign_bond_conjugation: single + conjugated => output bond order 2.
    """
    type_by_idx = {a.index: a.atom_type for a in assignments}

    def _is_aromatic_ring(ring: tuple[int, ...]) -> bool:
        """Return whether all atoms in a ring are aromatic-classified."""
        return all(idx in state.atms_aro for idx in ring)

    def _is_ring_ncnh(i: int, j: int) -> bool:
        """Return whether a bond matches Rosetta's ring(N=C)-N-H exception."""
        # Rosetta special case: [ring N=C]-N-H.
        i_aro = i in state.atms_aro
        j_aro = j in state.atms_aro
        if i_aro == j_aro:
            return False
        ring_idx, branch_idx = (i, j) if i_aro else (j, i)
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        branch_atom = mol.GetAtomWithIdx(branch_idx)
        if ring_atom.GetAtomicNum() != 6 or branch_atom.GetAtomicNum() != 7:
            return False
        _, nH_branch, _, _, _, _ = _neighbor_counts(branch_atom, state)
        if nH_branch < 1:
            return False

        ring_ids = state.ring_membership_by_idx.get(ring_idx, set())
        for nbr in ring_atom.GetNeighbors():
            k = nbr.GetIdx()
            if k == branch_idx:
                continue
            if nbr.GetAtomicNum() != 7 or _get_hyb(nbr, state) == HYB_SP3:
                continue
            if k not in state.atms_aro:
                continue
            if ring_ids & state.ring_membership_by_idx.get(k, set()):
                return True
        return False

    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)
        if ai.GetAtomicNum() == 1 or aj.GetAtomicNum() == 1:
            continue

        # SetupTopology: skip if either endpoint is sp3.
        hi = _get_hyb(ai, state)
        hj = _get_hyb(aj, state)
        if hi == HYB_SP3 or hj == HYB_SP3:
            continue

        ti = type_by_idx.get(i, "")
        tj = type_by_idx.get(j, "")
        if ti not in _CONJUGATING_ATOM_CLASSES or tj not in _CONJUGATING_ATOM_CLASSES:
            continue

        ring_ids_i = state.ring_membership_by_idx.get(i, set())
        ring_ids_j = state.ring_membership_by_idx.get(j, set())
        shared_ring_ids = ring_ids_i & ring_ids_j

        # Rosetta: skip conjugation when the bond belongs to a non-aromatic
        # (puckering/strained) ring.
        ring_puckering = False
        for rid in shared_ring_ids:
            if rid >= len(state.rings):
                continue
            if not _is_aromatic_ring(state.rings[rid]):
                ring_puckering = True
                break
        if ring_puckering:
            continue

        # Skip bonds in rings larger than 6.
        min_ring = 100
        for rid in shared_ring_ids:
            if rid < len(state.rings):
                min_ring = min(min_ring, len(state.rings[rid]))
        if min_ring > 6 and min_ring < 100:
            continue

        # Rosetta biaryl/special pivots: avoid forcing conjugation across
        # ring-to-ring aromatic junctions except the ring(N=C)-N-H exception.
        if not shared_ring_ids and (i in state.atms_aro or j in state.atms_aro):
            i_in_ring = bool(state.ring_membership_by_idx.get(i, set()))
            j_in_ring = bool(state.ring_membership_by_idx.get(j, set()))
            if i_in_ring and j_in_ring and not _is_ring_ncnh(i, j):
                continue

        bond.SetBondType(Chem.BondType.DOUBLE)
        bond.SetIsAromatic(False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_tmol_atom_types(mol: Chem.Mol) -> list[AtomTypeAssignment]:
    """Assign Rosetta generic_potential atom types to each atom in a Mol.

    Follows the exact classification logic from Rosetta's AtomTypeClassifier
    (mol2genparams), including the polar-carbon modifier and ring-nitrogen
    corrections. Atom names follow Rosetta's rename_atoms convention:
    heavy atoms as <Element><count>, hydrogens as H<bonded_element><count>.
    """
    mol = _prepare_mol_for_typing(mol)
    mol = _ensure_explicit_hydrogens(mol)
    state = _build_rosetta_typing_state(mol)

    classifiers = {
        1: _classify_H,
        6: _classify_C,
        7: _classify_N,
        8: _classify_O,
        9: _classify_halogen,
        15: _classify_P,
        16: _classify_S,
        17: _classify_halogen,
        35: _classify_halogen,
        53: _classify_halogen,
    }

    # Pass 1: classify all atoms, name heavy atoms
    heavy_assignments: list[tuple[int, str, str, int]] = []
    h_atoms: list[tuple[int, int, str]] = []  # (idx, bonded_heavy_idx, atom_type)
    elem_counts: dict[str, int] = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()

        classifier = classifiers.get(z)
        if classifier is not None:
            atom_type = classifier(atom, mol, state)
        else:
            elem = _elem_symbol(z)
            atom_type = "CS"
            logger.warning("Unknown element %s (Z=%d), defaulting to CS", elem, z)

        if z == 1:
            bonded_heavy_idx = -1
            for nbr in atom.GetNeighbors():
                bonded_heavy_idx = nbr.GetIdx()
                break
            h_atoms.append((idx, bonded_heavy_idx, atom_type))
        else:
            elem = _elem_symbol(z)
            elem_counts[elem] = elem_counts.get(elem, 0) + 1
            atom_name = f"{elem}{elem_counts[elem]}"
            heavy_assignments.append((idx, atom_name, atom_type, z))

    # Build heavy atom index->element mapping for hydrogen naming
    heavy_elem_by_idx = {idx: _elem_symbol(z) for idx, _, _, z in heavy_assignments}

    # Pass 2: name hydrogens as H<bonded_heavy_element><count> — matches
    # Rosetta mol2genparams's classic naming. Note that some other
    # ground-truth sources (e.g. Frank's .tmol files) use sequential
    # H1/H2/... naming instead; the chemistry tests collapse this via
    # graph-based parent matching rather than byte-equal H names.
    h_name_counts: dict[str, int] = {}
    h_assignments: list[tuple[int, str, str]] = []
    for h_idx, heavy_idx, h_type in h_atoms:
        heavy_elem = heavy_elem_by_idx.get(heavy_idx, "")
        h_prefix = f"H{heavy_elem}"
        h_name_counts[h_prefix] = h_name_counts.get(h_prefix, 0) + 1
        h_name = f"{h_prefix}{h_name_counts[h_prefix]}"
        h_assignments.append((h_idx, h_name, h_type))

    # Merge and sort by index (heavy first, then H, as Rosetta does)
    assignments = []
    for idx, name, atype, z in heavy_assignments:
        assignments.append(
            AtomTypeAssignment(
                atom_name=name,
                atom_type=atype,
                element=_elem_symbol(z),
                index=idx,
            )
        )
    for idx, name, atype in h_assignments:
        assignments.append(
            AtomTypeAssignment(
                atom_name=name,
                atom_type=atype,
                element="H",
                index=idx,
            )
        )

    # Pass 3: modify polar carbons
    assignments = _modify_polar_c(assignments, mol)

    # Pass 4: Rosetta amide bond order correction (Nad/Nad3-CDp)
    _correct_amide_bond_orders(assignments, mol)

    # Pass 5: ring nitrogen corrections
    assignments = _correct_ring_nitrogen(assignments, mol, state)

    # Pass 6: Rosetta conjugation bond-order output correction
    _correct_conjugated_single_bond_orders(assignments, mol, state)

    return assignments
