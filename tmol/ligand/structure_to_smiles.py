"""Derive a ligand SMILES string from a biotite ``AtomArray``.

This is the entry to tmol's unified ligand path: a CIF/atom-array ligand is
converted to a SMILES string here, then handed to the existing
SMILES -> params pipeline (:func:`nonstandard_residue_info_from_smiles_via_mol2`).

The SMILES always reflects the *input atoms as given* -- there is no residue-code
/ CCD-template lookup (that risks substituting an unrelated molecule when a CIF
uses a generic residue code such as ``LG1``). Candidates are derived in priority
order:

1. **Existing bonds** — when the ``AtomArray`` already carries a bond table
   (e.g. a CIF with a ``_chem_comp_bond`` block).
2. **Geometry** — bond perception from 3D coordinates via RDKit
   ``rdDetermineBonds`` (with the vendored ``xyz2mol_tm`` transition-metal
   fallback), for bonds-absent inputs.

Both route through the vendored atomworks :func:`atom_array_to_rdkit`.
"""

from __future__ import annotations

import logging

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from tmol.ligand.external.atomworks_rdkit import atom_array_to_rdkit

logger = logging.getLogger(__name__)


def _has_bonds(atom_array: AtomArray) -> bool:
    """Return True if the AtomArray carries a non-empty bond table."""
    return atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0


def _system_charge(atom_array: AtomArray, system_charge: int | None) -> int:
    """Resolve the net formal charge to use for geometry-based bond perception."""
    if system_charge is not None:
        return int(system_charge)
    if "charge" in atom_array.get_annotation_categories():
        return int(np.nansum(atom_array.charge))
    return 0


def _mol_to_smiles(mol: Chem.Mol) -> str | None:
    """Canonical heavy-atom SMILES for a Mol, or None if it can't be produced."""
    try:
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        logger.debug("MolToSmiles failed", exc_info=True)
        return None
    return smiles or None


def ligand_smiles_candidates_from_atom_array(
    atom_array: AtomArray,
    *,
    res_name: str | None = None,
    system_charge: int | None = None,
) -> list[str]:
    """Return candidate SMILES for a ligand AtomArray, best source first.

    The SMILES is derived purely from the input atoms (never a residue-code /
    CCD-template lookup). The list is de-duplicated (by canonical SMILES) and
    ordered: existing-bonds, then geometry. Callers can try each in turn.

    Args:
        atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
        res_name: Residue code, used only for log messages.
        system_charge: Net formal charge for geometry-based bond perception.
            Defaults to the summed ``charge`` annotation, else 0.

    Returns:
        Ordered, de-duplicated list of candidate SMILES strings (possibly empty).
    """
    candidates: list[str] = []

    def _add(smiles: str | None) -> None:
        if smiles and smiles not in candidates:
            candidates.append(smiles)

    # 1. Existing bonds from the AtomArray.
    if _has_bonds(atom_array):
        try:
            mol = atom_array_to_rdkit(
                atom_array, infer_bonds=False, hydrogen_policy="remove"
            )
            _add(_mol_to_smiles(mol))
        except Exception:
            logger.debug(
                "Existing-bonds SMILES derivation failed for %s",
                res_name,
                exc_info=True,
            )

    # 2. Geometry-based bond perception.
    try:
        charge = _system_charge(atom_array, system_charge)
        mol = atom_array_to_rdkit(
            atom_array,
            infer_bonds=True,
            system_charge=charge,
            hydrogen_policy="remove",
        )
        _add(_mol_to_smiles(mol))
    except Exception:
        logger.debug(
            "Geometry SMILES derivation failed for %s", res_name, exc_info=True
        )

    return candidates


def ligand_smiles_from_atom_array(
    atom_array: AtomArray,
    *,
    res_name: str | None = None,
    system_charge: int | None = None,
) -> str:
    """Derive the best-available SMILES for a ligand AtomArray.

    Tries the existing-bonds then geometry routes (see
    :func:`ligand_smiles_candidates_from_atom_array`) and returns the first
    that succeeds. Never does a residue-code / CCD-template lookup.

    Args:
        atom_array: The ligand sub-array.
        res_name: Residue code, used only for log/error messages.
        system_charge: Net formal charge for geometry-based bond perception.

    Returns:
        A canonical SMILES string.

    Raises:
        ValueError: If no route produces a SMILES.
    """
    candidates = ligand_smiles_candidates_from_atom_array(
        atom_array, res_name=res_name, system_charge=system_charge
    )
    if not candidates:
        raise ValueError(
            f"Could not derive a SMILES for ligand "
            f"{res_name or '<unknown>'} (no usable bonds or geometry)."
        )
    return candidates[0]
