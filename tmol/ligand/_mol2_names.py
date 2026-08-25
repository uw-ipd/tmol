"""Rosetta-style disambiguation of duplicate Tripos atom names in MOL2 files.

Rosetta ``mol2genparams`` renames repeated atom names in the same residue
(e.g. a second ``C2'`` becomes ``C2'2``). PLI fixtures such as ``fgfr1`` rely
on this when the same label appears on distinct atoms in one mol2 block.
"""

from __future__ import annotations

from rdkit import Chem


def disambiguate_mol2_atom_name(name: str, occurrence: int) -> str:
    """Return the Rosetta/mol2gen name for the ``occurrence``-th use of ``name``.

    The first occurrence keeps the original name; the second becomes
    ``{name}2``, the third ``{name}3``, etc.
    """
    if occurrence <= 1:
        return name
    return f"{name}{occurrence}"


def _raw_tripos_atom_name(atom: Chem.Atom, fallback_index: int) -> str:
    """Return the atom's raw Tripos name, or ``<symbol><fallback_index+1>``."""
    if atom.HasProp("_TriposAtomName"):
        name = atom.GetProp("_TriposAtomName").strip()
        if name:
            return name
    return f"{atom.GetSymbol()}{fallback_index + 1}"


def apply_disambiguated_mol2_names(mol: Chem.Mol) -> list[str]:
    """Assign unique ``_TriposAtomName`` values on ``mol`` (mol2gen convention).

    Names are assigned in RDKit atom-index order, matching the order of the
    ``@<TRIPOS>ATOM`` block in the source file.

    Returns:
        The disambiguated name for each atom index.
    """
    seen: dict[str, int] = {}
    names: list[str] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        raw = _raw_tripos_atom_name(atom, idx)
        seen[raw] = seen.get(raw, 0) + 1
        name = disambiguate_mol2_atom_name(raw, seen[raw])
        atom.SetProp("_TriposAtomName", name)
        names.append(name)
    return names
