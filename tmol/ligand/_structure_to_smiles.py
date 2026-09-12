"""Derive a ligand SMILES string from a biotite ``AtomArray``.

This is the entry to tmol's unified ligand path: a CIF/atom-array ligand is
converted to a SMILES string here, then handed to the existing
SMILES -> params pipeline (:func:`nonstandard_residue_info_from_smiles_via_mol2`).

The SMILES always reflects the *input atoms as given* -- there is no residue-code
/ CCD-template lookup (that risks substituting an unrelated molecule when a CIF
uses a generic residue code such as ``L_1``).

Bond orders must come from the input: the ``AtomArray`` is required to carry a
bond table (e.g. a CIF with a ``_chem_comp_bond`` block, or a mol2 BOND
section). We deliberately do *not* perceive bonds from 3D geometry -- a
bonds-absent input (such as a plain PDB ligand) is a hard error, because guessed
bond orders would silently corrupt the generated params database.

The SMILES is built with the shared ligand builder
:func:`tmol.ligand._rdkit_mol.rdkit_mol_from_ligand_atom_array` -- the same
AtomArray -> RDKit path the params pipeline uses -- so the derived SMILES and
the prepared molecule always agree on chemistry.
"""

from __future__ import annotations

import logging

from biotite.structure import AtomArray
from rdkit import Chem
from atomworks.io.tools.protonation import (
    correct_carboxylate_bond_orders as apply_geometry_bond_corrections,
    _infer_carboxylate_bonds,
    _sp2_angle_sum,
)

from tmol.ligand._rdkit_mol import rdkit_mol_from_ligand_atom_array

logger = logging.getLogger(__name__)


def _has_bonds(atom_array: AtomArray) -> bool:
    """Return True if the AtomArray carries a non-empty bond table."""
    return atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0


def _mol_to_smiles(mol: Chem.Mol) -> str | None:
    """Canonical heavy-atom SMILES for a Mol, or None if it can't be produced."""
    try:
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        logger.debug("MolToSmiles failed", exc_info=True)
        return None
    return smiles or None


def _tag_source_atom_map(mol: Chem.Mol, atom_array: AtomArray) -> None:
    """Tag each heavy atom with map number = source index + 1 (rides the SMILES
    through the mol2 pipeline for atom naming). No-op if counts disagree."""
    heavy_idx = [i for i, e in enumerate(atom_array.element) if str(e) != "H"]
    if mol.GetNumAtoms() != len(heavy_idx):
        return
    for j in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(j).SetAtomMapNum(int(heavy_idx[j]) + 1)


def ligand_smiles_from_atom_array(
    atom_array: AtomArray,
    *,
    res_name: str | None = None,
    with_atom_map: bool = False,
) -> str:
    """Derive a canonical SMILES for a ligand AtomArray from its bond table.

    The SMILES is derived purely from the input atoms and their explicit bonds
    (never a residue-code / CCD-template lookup, never geometry-based bond
    perception). Geometry-based bond-*order* corrections are still applied for
    motifs the input encodes inconsistently (carboxylates).

    Args:
        atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
        res_name: Residue code, used only for log/error messages.
        with_atom_map: Tag heavy atoms with source-index map numbers for CIF
            atom naming downstream.

    Returns:
        A canonical SMILES string.

    Raises:
        ValueError: If the AtomArray carries no bond table (bond orders must be
            supplied by the input; a bonds-absent ligand such as a plain PDB
            cannot be prepared without guessing chemistry), or if no SMILES
            could be derived from the bonds present.
    """
    label = res_name or "<unknown>"
    if not _has_bonds(atom_array):
        raise ValueError(
            f"Ligand {label} has no bond table; bond orders are required to "
            "derive a SMILES. Supply an input with explicit bonds (CIF "
            "_chem_comp_bond block, mol2, or SMILES) -- bond perception from 3D "
            "geometry is intentionally disabled."
        )

    # Build from the explicit bonds
    #  -> then apply geometry bond-order corrections for common "mistakes"
    def _attempt(repair_chemistry: bool) -> str | None:
        mol = rdkit_mol_from_ligand_atom_array(
            atom_array,
            res_name=res_name or "ligand",
            repair_chemistry=repair_chemistry,
        )
        mol = apply_geometry_bond_corrections(mol)
        if with_atom_map:
            _tag_source_atom_map(mol, atom_array)
        return _mol_to_smiles(mol)

    # Pass 1: try to convert exactly
    # If Pass 1 fails, make Pass 2: apply rules to recover
    smiles = None
    try:
        smiles = _attempt(repair_chemistry=False)
    except Exception:
        logger.debug(
            "SMILES derivation (no repair) failed for %s", res_name, exc_info=True
        )

    if not smiles:
        try:
            smiles = _attempt(repair_chemistry=True)
        except Exception as err:
            logger.debug(
                "SMILES derivation (repair) failed for %s", res_name, exc_info=True
            )
            raise ValueError(
                f"Could not derive a SMILES for ligand {label} from its bond table."
            ) from err

    if not smiles:
        raise ValueError(
            f"Could not derive a SMILES for ligand {label} from its bond table."
        )
    return smiles
