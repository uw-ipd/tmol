"""Derive a ligand SMILES string from a biotite ``AtomArray``.

This is the entry to tmol's unified ligand path: a CIF/atom-array ligand is
converted to a SMILES string here, then handed to the existing
SMILES -> params pipeline (:func:`nonstandard_residue_info_from_smiles_via_mol2`).

The SMILES always reflects the *input atoms as given* -- there is no residue-code
/ CCD-template lookup (that risks substituting an unrelated molecule when a CIF
uses a generic residue code such as ``LG1``).

Bond orders must come from the input: the ``AtomArray`` is required to carry a
bond table (e.g. a CIF with a ``_chem_comp_bond`` block, or a mol2 BOND
section). We deliberately do *not* perceive bonds from 3D geometry -- a
bonds-absent input (such as a plain PDB ligand) is a hard error, because guessed
bond orders would silently corrupt the generated params database.

The SMILES is built with the shared ligand builder
:func:`tmol.ligand.rdkit_mol.rdkit_mol_from_ligand_atom_array` -- the same
AtomArray -> RDKit path the params pipeline uses -- so the derived SMILES and
the prepared molecule always agree on chemistry.
"""

from __future__ import annotations

import logging

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from tmol.ligand.rdkit_mol import rdkit_mol_from_ligand_atom_array

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


# Below this, a C-O bond reads as double/delocalized rather than a hydroxyl
# single bond (carboxylate ~1.25 A, C=O ~1.21, C-OH ~1.31, diol C-O ~1.41).
_CARBOXYL_CO_MAX = 1.36
# Sum of the three bond angles at an sp2 (planar) carbon is 360; sp3 ~328.5.
_SP2_ANGLE_SUM_MIN = 355.0


def _sp2_angle_sum(conf, center: int, neighbors: list[int]) -> float | None:
    """Sum of the three bond angles at ``center`` (deg); None if degenerate."""
    cpos = np.asarray(conf.GetAtomPosition(center))
    vecs = [np.asarray(conf.GetAtomPosition(n)) - cpos for n in neighbors]
    if any(np.linalg.norm(v) == 0 for v in vecs):
        return None
    units = [v / np.linalg.norm(v) for v in vecs]
    total = 0.0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            total += np.degrees(
                np.arccos(np.clip(np.dot(units[i], units[j]), -1.0, 1.0))
            )
    return total


def _infer_carboxylate_bonds(rw: Chem.RWMol, conf) -> int:
    """Correct carboxylates mis-encoded as geminal diols; return #corrected.

    A carbon bonded to exactly two terminal oxygens whose geometry is planar
    with short C-O bonds is a delocalized carboxylate, not a diol. Some inputs
    (CIFs with SING/SING C-O, mol2s with non-ring ``ar`` bonds) drop the double
    bond, so the derived SMILES protonates both oxygens. Rewrite each such
    center to ``C(=O)[O-]`` from the input geometry.
    """
    n_fixed = 0
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 3:
            continue
        c = atom.GetIdx()
        term_os = [
            nb.GetIdx()
            for nb in atom.GetNeighbors()
            if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1
        ]
        if len(term_os) != 2:
            continue
        cpos = np.asarray(conf.GetAtomPosition(c))
        co_dists = [
            float(np.linalg.norm(np.asarray(conf.GetAtomPosition(o)) - cpos))
            for o in term_os
        ]
        if any(d > _CARBOXYL_CO_MAX for d in co_dists):
            continue
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        angle_sum = _sp2_angle_sum(conf, c, nbrs)
        if angle_sum is None or angle_sum < _SP2_ANGLE_SUM_MIN:
            continue

        oa, ob = term_os
        for idx in (c, oa, ob):
            rw.GetAtomWithIdx(idx).SetIsAromatic(False)
        b_oa = rw.GetBondBetweenAtoms(c, oa)
        b_ob = rw.GetBondBetweenAtoms(c, ob)
        b_oa.SetIsAromatic(False)
        b_ob.SetIsAromatic(False)
        b_oa.SetBondType(Chem.BondType.DOUBLE)
        b_ob.SetBondType(Chem.BondType.SINGLE)
        # Reset both O charges
        rw.GetAtomWithIdx(oa).SetFormalCharge(0)
        rw.GetAtomWithIdx(ob).SetFormalCharge(-1)
        n_fixed += 1
        logger.info("inferring COO- from geometry (carbon atom idx %d)", c)
    return n_fixed


def apply_geometry_bond_corrections(mol: Chem.Mol) -> Chem.Mol:
    """Repair input bond orders that disagree with the 3D geometry.

    Runs each geometry-based correction rule (carboxylate only, for now) and
    re-sanitizes. Returns the input unchanged when there is no conformer or no
    correction applies. More rules (nitro, phosphate, sulfonate, ...) can be
    added as separate ``_infer_*`` functions and dispatched here.
    """
    if mol.GetNumConformers() == 0:
        return mol
    rw = Chem.RWMol(mol)
    conf = rw.GetConformer()
    n_fixed = _infer_carboxylate_bonds(rw, conf)
    if n_fixed == 0:
        return mol
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        logger.debug("geometry bond correction failed to sanitize", exc_info=True)
        return mol
    return out


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

    # Build from the explicit bonds, then apply geometry bond-order corrections
    # for motifs the input encodes inconsistently (carboxylates).
    try:
        mol = rdkit_mol_from_ligand_atom_array(
            atom_array, res_name=res_name or "ligand"
        )
        mol = apply_geometry_bond_corrections(mol)
        if with_atom_map:
            _tag_source_atom_map(mol, atom_array)
        smiles = _mol_to_smiles(mol)
    except Exception as err:
        logger.debug("SMILES derivation failed for %s", res_name, exc_info=True)
        raise ValueError(
            f"Could not derive a SMILES for ligand {label} from its bond table."
        ) from err

    if not smiles:
        raise ValueError(
            f"Could not derive a SMILES for ligand {label} from its bond table."
        )
    return smiles
