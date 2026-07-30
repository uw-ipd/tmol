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

    # 1. Existing bonds from the AtomArray, with geometry-based bond-order
    #    corrections for motifs the input encodes inconsistently (carboxylates).
    if _has_bonds(atom_array):
        try:
            mol = atom_array_to_rdkit(
                atom_array, infer_bonds=False, hydrogen_policy="remove"
            )
            mol = apply_geometry_bond_corrections(mol)
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
