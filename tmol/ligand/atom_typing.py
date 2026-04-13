"""Atom type assignment for ligand atoms.

Assigns Rosetta generic_potential atom types to atoms in an OpenBabel OBMol.
The classification logic is a faithful port of Rosetta's AtomTypeClassifier
(from mol2genparams / generic_potential) and produces identical atom types
and atom names, including the polar-carbon modifier and the Rosetta hydrogen
naming convention (H<bonded_element><count>).
"""

import logging
from typing import NamedTuple

from openbabel import openbabel

from tmol.ligand.chemistry_tables import get_hbond_properties, get_polar_classes

logger = logging.getLogger(__name__)

POLARCLASSES = get_polar_classes()

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

# Database-backed HBond properties keyed by atom type name.
HBOND_PROPERTIES: dict[str, dict] = get_hbond_properties()


class AtomTypeAssignment(NamedTuple):
    atom_name: str
    atom_type: str
    element: str
    index: int


def _elem_symbol(atomic_num: int) -> str:
    sym = ELEMENT_SYMBOLS.get(atomic_num)
    if sym is not None:
        return sym
    return openbabel.GetSymbol(atomic_num)


def _is_hydrogen(obatom: openbabel.OBAtom) -> bool:
    return obatom.GetAtomicNum() == 1


def _ensure_explicit_hydrogens(obmol: openbabel.OBMol) -> None:
    """Add explicit hydrogens when an OBMol has none.

    Rosetta's degree-based carbon typing assumes explicit H neighbors are
    present; without them, saturated carbons can be misclassified as CD/CT.
    """
    has_hydrogen = any(
        obatom.GetAtomicNum() == 1 for obatom in openbabel.OBMolAtomIter(obmol)
    )
    if not has_hydrogen:
        obmol.AddHydrogens()


def _neighbor_counts(obatom: openbabel.OBAtom) -> tuple[int, int, int, int, int, int]:
    """Return (nC, nH, nO, nN, nS, ntot) for neighbors of obatom."""
    nC = nH = nO = nN = nS = 0
    ntot = 0
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
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


def _get_hyb(obatom: openbabel.OBAtom) -> int:
    """Map openbabel hybridization to Rosetta mol2 convention.

    OB: 1=sp, 2=sp2, 3=sp3, IsAromatic() separate.
    Rosetta mol2: 1=sp, 2=sp2, 3=sp3, 8=amide, 9=aromatic.
    """
    if obatom.IsAromatic():
        return 9
    hyb = obatom.GetHyb()
    if hyb == 0:
        return 3
    return hyb


def _has_sp2_double_bonded_O(obatom: openbabel.OBAtom) -> bool:
    """Check if a carbon has a double-bonded sp2 oxygen (C=O)."""
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        if nbr.GetAtomicNum() == 8 and bond.GetBondOrder() == 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-element classifiers — ported 1:1 from Rosetta AtomTypeClassifier
# ---------------------------------------------------------------------------


def _classify_H(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
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


def _classify_C(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    hyb = _get_hyb(obatom)
    nbonds = obatom.GetTotalDegree()
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(obatom)

    if hyb == 9:
        prefix = "CR"
    elif obatom.IsChiral():
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


def _classify_N_hetero(obatom, hyb, nC, nH, nO, nN, ntot):
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
            return "Nad3"
        if nH == 0:
            return "NG2"
        if nH == 1:
            for bond in openbabel.OBAtomBondIter(obatom):
                nbr = bond.GetNbrAtom(obatom)
                if nbr.GetAtomicNum() == 6 and _get_hyb(nbr) == 2:
                    if _has_sp2_double_bonded_O(nbr):
                        return "Nad"
            return "NG21"
        return "NG22"

    return "NG2"


def _classify_N_sp2(obatom, nC, nH):
    """Classify sp2 nitrogen connected only to C/H."""
    is_guanidinium = False
    is_amide = False
    nCaro = 0

    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        if nbr.GetAtomicNum() != 6:
            continue
        nbr_hyb = _get_hyb(nbr)
        if nbr_hyb in (2, 9):
            nCaro += 1
            nOsp2_j = 0
            nNsp2_j = 0
            for bond_k in openbabel.OBAtomBondIter(nbr):
                nbr_k = bond_k.GetNbrAtom(nbr)
                if nbr_k.GetIndex() == obatom.GetIndex():
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
        if nH == 1 and obatom.IsAromatic():
            return "Nin"
        return "NG21" if nH == 1 else "Nim" if nH == 0 else "NG22"

    if nC == 3 and nCaro >= 1:
        return "Nad3"

    if nH == 0:
        return "NG2"
    return "NG21" if nH == 1 else "NG22"


def _classify_N(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    hyb = _get_hyb(obatom)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(obatom)

    if hyb == 1:
        return "NG1"

    if hyb == 2 and ntot == 4 and nH == 3:
        return "Nam"
    if hyb == 2 and ntot == 4 and 1 <= nH < 3:
        return "Nam2"

    if nC + nH < ntot:
        return _classify_N_hetero(obatom, hyb, nC, nH, nO, nN, ntot)

    if hyb == 2:
        return _classify_N_sp2(obatom, nC, nH)

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


def _classify_O_no_carbon(obatom, hyb, nH, nN, ntot):
    """Classify oxygen not bonded to any carbon."""
    if hyb == 3:
        if nH >= 1:
            is_PO4H = False
            if ntot == 2:
                for bond in openbabel.OBAtomBondIter(obatom):
                    if bond.GetNbrAtom(obatom).GetAtomicNum() == 15:
                        is_PO4H = True
                        break
            return "Ohx" if is_PO4H else "OG31"
        return "OG3"
    if hyb == 2:
        return "Ont" if nN == 1 and ntot == 1 else "OG2"
    return "OG2"


def _classify_O_sp2(obatom, nC):
    """Classify sp2 oxygen bonded to at least one carbon."""
    if nC == 2:
        return "Ofu"

    c_nbr = None
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
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
        for bond in openbabel.OBAtomBondIter(c_nbr):
            nbr = bond.GetNbrAtom(c_nbr)
            if nbr.GetIndex() == obatom.GetIndex():
                continue
            if nbr.GetAtomicNum() == 8:
                deg = nbr.GetTotalDegree()
                if deg == 2:
                    return "Oal"
                if deg == 1:
                    return "Oat"
                return "OG2"

    return "OG2"


def _classify_O(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    hyb = _get_hyb(obatom)
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(obatom)

    if nC == 0:
        return _classify_O_no_carbon(obatom, hyb, nH, nN, ntot)

    if ntot > 2:
        return "OG2"

    if hyb == 3:
        if nH >= 1:
            return "Ohx"
        if ntot == 2 and nC == 2:
            return "Ofu" if obatom.IsAromatic() else "Oet"
        return "OG31" if nH >= 1 else "OG3"

    if hyb == 2:
        return _classify_O_sp2(obatom, nC)

    return "OG2"


def _classify_S(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    nC, nH, nO, nN, nS, ntot = _neighbor_counts(obatom)
    if nC == 1 and nH == 1 and ntot == 2:
        return "Sth"
    elif nC + nS == 2 and ntot == 2:
        return "SR" if obatom.IsAromatic() else "Ssl"
    elif ntot == 1:
        return "SG2"
    else:
        hyb = _get_hyb(obatom)
        return "SG5" if hyb == 2 and ntot >= 4 else "SG3"


def _classify_P(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    if obatom.GetTotalDegree() >= 4:
        return "PG5"
    return "PG3"


def _classify_halogen(obatom: openbabel.OBAtom, obmol: openbabel.OBMol) -> str:
    z = obatom.GetAtomicNum()
    base = {9: "F", 17: "Cl", 35: "Br", 53: "I"}[z]
    if obatom.GetTotalDegree() == 1:
        for bond in openbabel.OBAtomBondIter(obatom):
            nbr = bond.GetNbrAtom(obatom)
            if nbr.GetAtomicNum() == 6 and nbr.IsAromatic():
                return base + "R"
    return base


# ---------------------------------------------------------------------------
# modify_polarC — post-classification pass matching Rosetta exactly
# ---------------------------------------------------------------------------


def _modify_polar_c(
    assignments: list[AtomTypeAssignment],
    obmol: openbabel.OBMol,
) -> list[AtomTypeAssignment]:
    """Convert CS/CD/CR/CT -> CSp/CDp/CRp/CTp for polar-adjacent carbons."""
    type_by_idx = {a.index: a.atom_type for a in assignments}

    result = []
    for a in assignments:
        obatom = obmol.GetAtom(a.index + 1)  # OB is 1-indexed
        if obatom.GetAtomicNum() != 6:
            result.append(a)
            continue

        atype = a.atom_type
        prefix = atype[:2]
        if prefix not in ("CS", "CD", "CR", "CT"):
            result.append(a)
            continue

        n_heavy = sum(
            1
            for bond in openbabel.OBAtomBondIter(obatom)
            if not _is_hydrogen(bond.GetNbrAtom(obatom))
        )
        n_total = obatom.GetTotalDegree()
        if n_total - (n_total - n_heavy) == 1:
            # aliphatic tip: only 1 heavy neighbor
            # n_heavy == 1
            pass
        if n_heavy <= 1:
            result.append(a)
            continue

        attached_to_polar = False
        for bond in openbabel.OBAtomBondIter(obatom):
            nbr = bond.GetNbrAtom(obatom)
            nbr_type = type_by_idx.get(nbr.GetIndex())
            if nbr_type in POLARCLASSES:
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
    obmol: openbabel.OBMol,
) -> list[AtomTypeAssignment]:
    """Correct Nad3->Nim for nitrogen in 5/6-membered rings."""
    result = list(assignments)

    sssr = obmol.GetSSSR()
    for ring in sssr:
        ring_size = ring.Size()
        if ring_size < 5 or ring_size > 6:
            continue

        for i, a in enumerate(result):
            obatom = obmol.GetAtom(a.index + 1)
            if obatom.GetAtomicNum() != 7:
                continue
            if not ring.IsMember(obatom):
                continue

            hyb = _get_hyb(obatom)
            if hyb not in (2, 9):
                continue

            nC, nH, nO, nN, nS, ntot = _neighbor_counts(obatom)
            if ntot < 3 and nH == 0 and nN >= 1:
                result[i] = a._replace(atom_type="Nim")

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_tmol_atom_types(
    obmol: openbabel.OBMol,
) -> list[AtomTypeAssignment]:
    """Assign Rosetta generic_potential atom types to each atom in an OBMol.

    Follows the exact classification logic from Rosetta's AtomTypeClassifier
    (mol2genparams), including the polar-carbon modifier and ring-nitrogen
    corrections.  Atom names follow Rosetta's rename_atoms convention:
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

    # Ensure degree-based atom typing sees explicit hydrogens, matching
    # legacy mol2genparams behavior for aliphatic carbons.
    _ensure_explicit_hydrogens(obmol)

    # Pass 1: classify all atoms, name heavy atoms
    heavy_assignments: list[tuple[int, str, str, int]] = []
    h_atoms: list[tuple[int, int]] = []  # (ob_index, bonded_heavy_ob_index)
    elem_counts: dict[str, int] = {}

    for obatom in openbabel.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        z = obatom.GetAtomicNum()

        classifier = classifiers.get(z)
        if classifier is not None:
            atom_type = classifier(obatom, obmol)
        else:
            elem = _elem_symbol(z)
            atom_type = "CS"
            logger.warning("Unknown element %s (Z=%d), defaulting to CS", elem, z)

        if z == 1:
            bonded_heavy_idx = -1
            for bond in openbabel.OBAtomBondIter(obatom):
                bonded_heavy_idx = bond.GetNbrAtom(obatom).GetIndex()
                break
            h_atoms.append((idx, bonded_heavy_idx, atom_type))
        else:
            elem = _elem_symbol(z)
            elem_counts[elem] = elem_counts.get(elem, 0) + 1
            atom_name = f"{elem}{elem_counts[elem]}"
            heavy_assignments.append((idx, atom_name, atom_type, z))

    # Build heavy atom index->element mapping for hydrogen naming
    heavy_elem_by_idx = {idx: _elem_symbol(z) for idx, _, _, z in heavy_assignments}

    # Pass 2: name hydrogens as H<bonded_heavy_element><count>
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
    assignments = _modify_polar_c(assignments, obmol)

    # Pass 4: ring nitrogen corrections
    assignments = _correct_ring_nitrogen(assignments, obmol)

    return assignments
