"""Atom type assignment for ligand atoms.

Maps element, bond order, hybridization, and connectivity information from
an OpenBabel OBMol to tmol/Rosetta atom types. The logic is ported from
Rosetta's molfile_to_params.py (assign_rosetta_types) and the
generic_potential mol2genparams type classifier.
"""

import logging
from typing import NamedTuple

from openbabel import openbabel

logger = logging.getLogger(__name__)


class AtomTypeAssignment(NamedTuple):
    """Result of atom type assignment for a single atom.

    Attributes:
        atom_name: Unique atom name (element + index, e.g. "C1", "O3").
        atom_type: The tmol atom type name (e.g. "CS1", "Ohx", "Hapo").
        element: Element symbol.
        index: 0-based atom index in the OBMol.
    """

    atom_name: str
    atom_type: str
    element: str
    index: int


def _count_heavy_neighbors(obatom: openbabel.OBAtom) -> int:
    """Count non-hydrogen neighbor atoms."""
    return sum(
        1
        for bond in openbabel.OBAtomBondIter(obatom)
        if not bond.GetNbrAtom(obatom).IsHydrogen()
    )


def _count_h_neighbors(obatom: openbabel.OBAtom) -> int:
    """Count hydrogen neighbor atoms."""
    return sum(
        1
        for bond in openbabel.OBAtomBondIter(obatom)
        if bond.GetNbrAtom(obatom).IsHydrogen()
    )


def _has_double_bond_to(obatom: openbabel.OBAtom, target_elem: int) -> bool:
    """Check if atom has a double bond to an atom of the given element."""
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        if bond.GetBondOrder() == 2 and nbr.GetAtomicNum() == target_elem:
            return True
    return False


def _has_any_double_bond(obatom: openbabel.OBAtom) -> bool:
    """Check if atom has any double bond."""
    return any(bond.GetBondOrder() == 2 for bond in openbabel.OBAtomBondIter(obatom))


def _is_in_ring(obatom: openbabel.OBAtom) -> bool:
    """Check if atom is in any ring."""
    return obatom.IsInRing()


def _is_aromatic(obatom: openbabel.OBAtom) -> bool:
    """Check if atom is aromatic."""
    return obatom.IsAromatic()


def _count_bonded_element(obatom: openbabel.OBAtom, elem: int) -> int:
    """Count neighbors of a specific element."""
    return sum(
        1
        for bond in openbabel.OBAtomBondIter(obatom)
        if bond.GetNbrAtom(obatom).GetAtomicNum() == elem
    )


def _assign_carbon_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for a carbon atom."""
    is_sat = not _has_any_double_bond(obatom) and not _is_aromatic(obatom)
    n_h = _count_h_neighbors(obatom)
    n_heavy = _count_heavy_neighbors(obatom)

    if is_sat:
        if n_h >= 3:
            return "CS3"
        elif n_h == 2:
            return "CS2"
        elif n_h == 1:
            return "CS1"
        else:
            return "CS"
    else:
        if _is_aromatic(obatom):
            if _is_in_ring(obatom):
                return "CR"
            return "CD"

        has_dbl_O = _has_double_bond_to(obatom, 8)
        has_dbl_N = _has_double_bond_to(obatom, 7)

        if has_dbl_O and _count_bonded_element(obatom, 8) >= 2:
            n_single_O = sum(
                1
                for bond in openbabel.OBAtomBondIter(obatom)
                if bond.GetNbrAtom(obatom).GetAtomicNum() == 8
                and bond.GetBondOrder() == 1
            )
            if n_single_O >= 1:
                return "CDp"

        if has_dbl_O and has_dbl_N:
            return "CDp"
        if has_dbl_O:
            return "CD"
        if has_dbl_N:
            return "CD1"

        if n_heavy >= 2:
            return "CD"

        return "CD2"


def _assign_nitrogen_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for a nitrogen atom."""
    n_heavy = _count_heavy_neighbors(obatom)
    n_h = _count_h_neighbors(obatom)
    total = n_heavy + n_h

    if _is_aromatic(obatom):
        if n_h >= 1:
            return "Nin"
        return "Nim"

    if _has_any_double_bond(obatom):
        if _has_double_bond_to(obatom, 6):
            bonded_c = [
                bond.GetNbrAtom(obatom)
                for bond in openbabel.OBAtomBondIter(obatom)
                if bond.GetNbrAtom(obatom).GetAtomicNum() == 6
                and bond.GetBondOrder() == 2
            ]
            for c in bonded_c:
                n_n_on_c = _count_bonded_element(c, 7)
                if n_n_on_c >= 3:
                    if n_h >= 2:
                        return "Ngu2"
                    return "Ngu1"

        if n_h >= 1:
            return "Nad"
        return "Nad3"

    has_amide_bond = False
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        if nbr.GetAtomicNum() == 6 and _has_double_bond_to(nbr, 8):
            has_amide_bond = True
            break

    if has_amide_bond:
        if n_h >= 2:
            return "Nam2"
        return "Nam"

    if total >= 4 or (n_heavy >= 3 and n_h >= 1):
        return "NG3"
    if n_h >= 2:
        return "NG22"
    if n_h >= 1:
        return "NG21"

    return "NG2"


def _assign_oxygen_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for an oxygen atom."""
    n_heavy = _count_heavy_neighbors(obatom)
    n_h = _count_h_neighbors(obatom)

    if _is_aromatic(obatom):
        return "Ofu"

    if _has_any_double_bond(obatom):
        if n_heavy >= 1:
            c_nbr = None
            for bond in openbabel.OBAtomBondIter(obatom):
                nbr = bond.GetNbrAtom(obatom)
                if nbr.GetAtomicNum() == 6 and bond.GetBondOrder() == 2:
                    c_nbr = nbr
                    break

            if c_nbr is not None:
                n_o_on_c = _count_bonded_element(c_nbr, 8)
                if n_o_on_c >= 2:
                    return "Oat"
                n_n_on_c = _count_bonded_element(c_nbr, 7)
                if n_n_on_c >= 1:
                    return "Oad"
                return "Oal"

            p_nbr = None
            for bond in openbabel.OBAtomBondIter(obatom):
                nbr = bond.GetNbrAtom(obatom)
                if nbr.GetAtomicNum() == 15:
                    p_nbr = nbr
                    break
            if p_nbr is not None:
                return "OG2"

        return "Oal"

    if n_h >= 1:
        return "Ohx"
    if n_heavy == 2:
        return "Oet"

    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        if nbr.GetAtomicNum() == 15:
            return "OG3"
        if nbr.GetAtomicNum() == 6:
            if _has_double_bond_to(nbr, 8):
                n_o_on_c = _count_bonded_element(nbr, 8)
                if n_o_on_c >= 2:
                    return "Oat"

    return "OG3"


def _assign_hydrogen_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for a hydrogen atom."""
    for bond in openbabel.OBAtomBondIter(obatom):
        nbr = bond.GetNbrAtom(obatom)
        elem = nbr.GetAtomicNum()
        if elem == 7 or elem == 16:
            return "HN"
        if elem == 8:
            return "HO"
        if elem == 6 and _is_aromatic(nbr):
            return "HR"
    return "HC"


def _assign_sulfur_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for a sulfur atom."""
    n_h = _count_h_neighbors(obatom)
    n_heavy = _count_heavy_neighbors(obatom)

    if _is_aromatic(obatom):
        return "SR"

    n_dbl_o = sum(
        1
        for bond in openbabel.OBAtomBondIter(obatom)
        if bond.GetNbrAtom(obatom).GetAtomicNum() == 8 and bond.GetBondOrder() == 2
    )
    if n_dbl_o >= 2:
        return "SG5"
    if n_dbl_o == 1:
        return "SG2"

    if n_h >= 1:
        return "Sth"
    if n_heavy == 2:
        return "Ssl"

    return "SG3"


def _assign_phosphorus_type(obatom: openbabel.OBAtom) -> str:
    """Assign a tmol atom type for a phosphorus atom."""
    n_o_bonds = _count_bonded_element(obatom, 8)
    n_heavy = _count_heavy_neighbors(obatom)

    if n_heavy >= 5:
        return "PG5"
    if n_o_bonds >= 3:
        return "PG5"
    return "PG3"


def assign_tmol_atom_types(
    obmol: openbabel.OBMol,
) -> list[AtomTypeAssignment]:
    """Assign tmol atom types to each atom in an OBMol.

    Uses element identity, bond orders, hybridization, ring membership,
    and neighbor connectivity to determine the appropriate Rosetta
    generic_potential atom type.

    Args:
        obmol: An OpenBabel OBMol with bonds and (optionally) 3D coords.

    Returns:
        A list of AtomTypeAssignment, one per atom in the molecule.
    """
    assignments: list[AtomTypeAssignment] = []
    elem_counts: dict[str, int] = {}

    for obatom in openbabel.OBMolAtomIter(obmol):
        atomic_num = obatom.GetAtomicNum()
        elem = openbabel.OBElements.GetSymbol(atomic_num)

        if atomic_num == 1:
            atom_type = _assign_hydrogen_type(obatom)
        elif atomic_num == 6:
            atom_type = _assign_carbon_type(obatom)
        elif atomic_num == 7:
            atom_type = _assign_nitrogen_type(obatom)
        elif atomic_num == 8:
            atom_type = _assign_oxygen_type(obatom)
        elif atomic_num == 9:
            atom_type = "FR"
        elif atomic_num == 15:
            atom_type = _assign_phosphorus_type(obatom)
        elif atomic_num == 16:
            atom_type = _assign_sulfur_type(obatom)
        elif atomic_num == 17:
            atom_type = "ClR"
        elif atomic_num == 35:
            atom_type = "BrR"
        elif atomic_num == 53:
            atom_type = "IR"
        else:
            atom_type = "CS"
            logger.warning(
                "Unknown element %s (Z=%d), defaulting to CS", elem, atomic_num
            )

        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        atom_name = f"{elem}{elem_counts[elem]}"

        assignments.append(
            AtomTypeAssignment(
                atom_name=atom_name,
                atom_type=atom_type,
                element=elem,
                index=obatom.GetIndex(),
            )
        )

    return assignments
