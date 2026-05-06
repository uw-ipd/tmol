"""Atom type assignment for ligand atoms.

Assigns Rosetta generic_potential atom types to atoms in an RDKit Mol.
The classification logic is a faithful port of Rosetta's AtomTypeClassifier
(from mol2genparams / generic_potential) and produces identical atom types
and atom names, including the polar-carbon modifier and the Rosetta hydrogen
naming convention (H<bonded_element><count>).
"""

import logging
from typing import NamedTuple

from rdkit import Chem

from tmol.ligand.chemistry_tables import get_polar_classes

logger = logging.getLogger(__name__)


# Rosetta hybridization convention (matches mol2genparams)
HYB_SP = 1
HYB_SP2 = 2
HYB_SP3 = 3
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


def _elem_symbol(atomic_num: int) -> str:
    sym = ELEMENT_SYMBOLS.get(atomic_num)
    if sym is not None:
        return sym
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)


def _is_hydrogen(atom: Chem.Atom) -> bool:
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
    from tmol.ligand.rdkit_mol import source_carried_kekule

    if source_carried_kekule(mol):
        ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
        try:
            Chem.SanitizeMol(mol, sanitizeOps=ops)
            return
        except Chem.rdchem.AtomValenceException:
            pass
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
        ops = (
            Chem.SANITIZE_ALL
            ^ Chem.SANITIZE_KEKULIZE
            ^ Chem.SANITIZE_SETAROMATICITY
            ^ Chem.SANITIZE_PROPERTIES
        )
        Chem.SanitizeMol(mol, sanitizeOps=ops)


_WAS_AROMATIC_PROP = "_tmol_was_aromatic"


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

    # Conditional kekulization: rings that carry an exocyclic C=O / C=N
    # (purine, cytosine, amide-substituted ring) were originally encoded
    # as sp2 in the source mol2 (``C.2`` → ``CD``), so we kekulize them
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


def _neighbor_counts(atom: Chem.Atom) -> tuple[int, int, int, int, int, int]:
    """Return (nC, nH, nO, nN, nS, ntot) for neighbors of atom."""
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


def _get_hyb(atom: Chem.Atom) -> int:
    """Map RDKit hybridization to Rosetta mol2 convention.

    Returns: 1=sp, 2=sp2, 3=sp3, 9=aromatic.
    Aromatic is checked first via GetIsAromatic(); other hybridizations
    map through _HYB_MAP with sp3 (3) as the default for unknown types.
    """
    if atom.GetIsAromatic():
        return HYB_AROMATIC
    return _HYB_MAP.get(atom.GetHybridization(), HYB_SP3)


def _has_sp2_double_bonded_O(atom: Chem.Atom) -> bool:
    """Check if a carbon has a double-bonded sp2 oxygen (C=O).

    Aromatic bonds are skipped (RDKit reports 1.5 for aromatic). A real
    C=O amide/acid/ester bond is non-aromatic with order 2.0.
    """
    for bond in atom.GetBonds():
        if bond.GetIsAromatic():
            continue
        nbr = bond.GetOtherAtom(atom)
        if nbr.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2.0:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-element classifiers — ported 1:1 from Rosetta AtomTypeClassifier
# ---------------------------------------------------------------------------


def _classify_H(atom: Chem.Atom, mol: Chem.Mol) -> str:
    for nbr in atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z == 6:
            return "HR" if _get_hyb(nbr) == 9 else "HC"
        elif z == 8:
            return "HO"
        elif z == 7:
            return "HN"
        elif z == 16:
            return "HS"
        else:
            return "HG"
    return "HG"


def _classify_C(atom: Chem.Atom, mol: Chem.Mol) -> str:
    from tmol.ligand.rdkit_mol import source_subtype

    hyb = _get_hyb(atom)
    nbonds = atom.GetDegree()
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)

    sub = source_subtype(atom)
    # mol2genparams takes the CR-vs-CD decision straight from the mol2
    # atom-type column. Only ``C.ar`` → ``CR``; everything else
    # (``C.2``, ``C.cat``, ``C.3`` even when RDKit aromatized the ring)
    # picks the Kekulé branch and gets ``CD``-flavoured types via the
    # degree-based rules below.
    if sub == "ar":
        prefix = "CR"
    elif sub:
        # Source explicit non-``.ar`` subtype — never treat as aromatic
        # for typing purposes, even if RDKit perceived the ring as
        # aromatic.
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

    if prefix in ("CS", "CD"):
        if nH > 0:
            return f"{prefix}{nH}"
        return prefix
    return prefix


def _classify_N_hetero(atom, hyb, nC, nH, nO, nN, ntot):
    """Classify nitrogen with non-C/H neighbors (lone pairs, heteroatoms)."""
    if hyb == 3:
        if ntot <= 3 and nH >= 1:
            return "Nam2"
        elif nH >= 1:
            return "Nam"
        return "NG3"

    if hyb in (2, 9):
        if nO >= 2:
            return "NG2"
        if nH == 0 and nN >= 1:
            # Azole-style ring N (triazole / tetrazole): all ring Ns
            # adjacent to another ring N get Nim regardless of degree.
            # The aromatic flag is read from the saved-pre-kekulize
            # snapshot since Kekulize clears the live one.
            if was_aromatic(atom):
                return "Nim"
            # Acyclic R2-N=C-... amide-style → Nad3 only when 3-coord;
            # 2-coord sp2 N with an N neighbor is an imine.
            if ntot == 3:
                return "Nad3"
            return "Nim"
        if nH == 0:
            return "NG2"
        if nH == 1:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6 and _get_hyb(nbr) == 2:
                    if _has_sp2_double_bonded_O(nbr):
                        return "Nad"
            return "NG21"
        return "NG22"

    return "NG2"


def _classify_N_sp2(atom, nC, nH):
    """Classify sp2 nitrogen connected only to C/H."""
    is_guanidinium = False
    is_amide = False
    nCaro = 0

    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        nbr_hyb = _get_hyb(nbr)
        if nbr_hyb in (2, 9):
            nCaro += 1
            nOsp2_j = 0
            nNsp2_j = 0
            for nbr_k in nbr.GetNeighbors():
                if nbr_k.GetIdx() == atom.GetIdx():
                    continue
                if nbr_k.GetAtomicNum() == 8 and _get_hyb(nbr_k) == 2:
                    nOsp2_j += 1
                elif nbr_k.GetAtomicNum() == 7 and _get_hyb(nbr_k) in (2, 9):
                    nNsp2_j += 1
            if nOsp2_j == 1:
                is_amide = True
            if nNsp2_j == 2:
                is_guanidinium = True

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
        # consult the saved aromaticity flag because the live one has
        # already been cleared by kekulize_tolerant by this point.
        if nH == 1 and was_aromatic(atom):
            return "Nin"
        return "NG21" if nH == 1 else "Nim" if nH == 0 else "NG22"

    if nC == 3 and nCaro >= 1:
        return "Nad3"

    if nH == 0:
        return "NG2"
    return "NG21" if nH == 1 else "NG22"


def _classify_N(atom: Chem.Atom, mol: Chem.Mol) -> str:
    from tmol.ligand.rdkit_mol import source_subtype

    hyb = _get_hyb(atom)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)
    sub = source_subtype(atom)

    # Source-mol2 ``N.am`` is an amide N → ``Nad`` (or ``Nad3`` when
    # 3-coord with no H). The carbon-aromaticity heuristic that drives
    # ``Nin`` is otherwise too eager when the neighboring C sits in an
    # aromatic ring (cytosine-like).
    if sub == "am":
        if ntot == 3 and nH == 0:
            return "Nad3"
        return "Nad"

    if hyb == 1:
        return "NG1"

    if hyb == 2 and ntot == 4 and nH == 3:
        return "Nam"
    if hyb == 2 and ntot == 4 and 1 <= nH < 3:
        return "Nam2"

    if nC + nH < ntot:
        return _classify_N_hetero(atom, hyb, nC, nH, nO, nN, ntot)

    if hyb == 2:
        return _classify_N_sp2(atom, nC, nH)

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


def _classify_O_no_carbon(atom, hyb, nH, nN, ntot):
    """Classify oxygen not bonded to any carbon."""
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
        return "Ont" if nN == 1 and ntot == 1 else "OG2"
    return "OG2"


def _classify_O_sp2(atom, nC):
    """Classify sp2 oxygen bonded to at least one carbon."""
    if nC == 2:
        # 2-carbon sp2 O is Ofu only when it sits in a furan-style
        # aromatic 5-ring. Acyclic ester / lactone Os that RDKit
        # aromatized via lone-pair conjugation get Oet.
        ri = atom.GetOwningMol().GetRingInfo()
        in_5ring = any(len(r) == 5 and atom.GetIdx() in r for r in ri.AtomRings())
        return "Ofu" if in_5ring else "Oet"

    c_nbr = None
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 6:
            c_nbr = nbr
            break
    if c_nbr is None:
        return "OG2"

    nC_j, nH_j, nO_j, nN_j, _, _ = _neighbor_counts(c_nbr)
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


def _classify_O(atom: Chem.Atom, mol: Chem.Mol) -> str:
    from tmol.ligand.rdkit_mol import source_subtype

    hyb = _get_hyb(atom)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)
    sub = source_subtype(atom)

    if nC == 0:
        return _classify_O_no_carbon(atom, hyb, nH, nN, ntot)

    if ntot > 2:
        return "OG2"

    # Source-mol2 ``O.2`` forces the sp2 branch even when the neighbour
    # carbon is aromatic. Rosetta's mol2genparams reads CR/CD and
    # Oal/Ohx straight from the source column; our phenolic-OH
    # short-circuit is a heuristic that only applies when the source
    # didn't already say sp2.
    if sub != "2" and ntot == 2 and nH >= 1 and nC == 1:
        c_nbr = next(n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_nbr.GetIsAromatic():
            return "Ohx"

    # When source says ``O.2``, route to the sp2 classifier directly.
    if sub == "2":
        return _classify_O_sp2(atom, nC)

    if hyb == 3:
        if nH >= 1:
            return "Ohx"
        if ntot == 2 and nC == 2:
            # Ofu = O sitting inside a furan-style aromatic 5-ring.
            # ``was_aromatic`` on its own is too permissive: RDKit's
            # sanitize also flags ester / lactone Os aromatic via lone-
            # pair conjugation with an adjacent carboxyl. We require
            # both the aromatic flag (pre-kekulize) AND membership in
            # a 5-membered ring.
            ri = mol.GetRingInfo()
            in_5ring = any(len(r) == 5 and atom.GetIdx() in r for r in ri.AtomRings())
            return "Ofu" if was_aromatic(atom) and in_5ring else "Oet"
        return "OG3"

    if hyb == 2 or hyb == 9:
        # Treat aromatic-perceived Os (hyb=9) the same as sp2 — RDKit
        # often marks carboxylate / nitro / heterocyclic oxygens
        # aromatic when their lone pair conjugates into a ring, but the
        # Rosetta classifier wants the Oat / Oal / Oad distinction
        # driven by the bonded carbon's substituent pattern.
        return _classify_O_sp2(atom, nC)

    return "OG2"


def _classify_S(atom: Chem.Atom, mol: Chem.Mol) -> str:
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)
    if nC == 1 and nH == 1 and ntot == 2:
        return "Sth"
    elif nC + nS == 2 and ntot == 2:
        return "SR" if atom.GetIsAromatic() else "Ssl"
    elif ntot == 1:
        return "SG2"
    else:
        hyb = _get_hyb(atom)
        return "SG5" if hyb == 2 and ntot >= 4 else "SG3"


def _classify_P(atom: Chem.Atom, mol: Chem.Mol) -> str:
    if atom.GetDegree() >= 4:
        return "PG5"
    return "PG3"


def _classify_halogen(atom: Chem.Atom, mol: Chem.Mol) -> str:
    z = atom.GetAtomicNum()
    base = {9: "F", 17: "Cl", 35: "Br", 53: "I"}[z]
    if atom.GetDegree() == 1:
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 6 and nbr.GetIsAromatic():
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
) -> list[AtomTypeAssignment]:
    """Override nitrogen type to Nim for sp2/aromatic N in 5/6-membered rings.

    Applies a graph-pattern match (ntot<3, nH==0, nN>=1) regardless of
    the current atom type -- any nitrogen matching the Rosetta ring-N
    pattern is forced to Nim.
    """
    result = list(assignments)

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

            hyb = _get_hyb(atom)
            if hyb not in (HYB_SP2, HYB_AROMATIC):
                continue

            nC, nH, nO, nN, nS, ntot = _neighbor_counts(atom)
            if ntot < 3 and nH == 0 and nN >= 1:
                result[i] = a._replace(atom_type="Nim")

    return result


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

    mol = _prepare_mol_for_typing(mol)
    mol = _ensure_explicit_hydrogens(mol)

    # Pass 1: classify all atoms, name heavy atoms
    heavy_assignments: list[tuple[int, str, str, int]] = []
    h_atoms: list[tuple[int, int, str]] = []  # (idx, bonded_heavy_idx, atom_type)
    elem_counts: dict[str, int] = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()

        classifier = classifiers.get(z)
        if classifier is not None:
            atom_type = classifier(atom, mol)
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

    # Pass 4: ring nitrogen corrections
    assignments = _correct_ring_nitrogen(assignments, mol)

    return assignments
