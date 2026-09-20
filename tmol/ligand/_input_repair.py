"""Repairs for source files whose charges and bond orders are incomplete.

These encode an input-cleanup policy for the formats TMol reads, not a general
bond perception method, and must only be requested for a source that needs them.
Chemistry reaches AtomWorks already repaired, so nothing downstream has to guess
what a file meant.
"""

import logging

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

# Below this, a C-O bond reads as double/delocalized rather than a hydroxyl
# single bond (carboxylate ~1.25 A, C=O ~1.21, C-OH ~1.31, diol C-O ~1.41).
_CARBOXYL_CO_MAX = 1.36
# Sum of the three bond angles at an sp2 (planar) carbon is 360; sp3 ~328.5.
_SP2_ANGLE_SUM_MIN = 355.0


def _sp2_angle_sum(
    conf: Chem.Conformer, center: int, neighbors: list[int]
) -> float | None:
    """Sum of the three bond angles at ``center`` (deg); None if degenerate."""
    cpos = np.asarray(conf.GetAtomPosition(center))
    vecs = [np.asarray(conf.GetAtomPosition(n)) - cpos for n in neighbors]
    norms = np.linalg.norm(vecs, axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        return None
    units = np.asarray(vecs) / norms[:, None]
    total = 0.0
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            total += np.degrees(
                np.arccos(np.clip(np.dot(units[i], units[j]), -1.0, 1.0))
            )
    return total


def _infer_carboxylate_bonds(rw: Chem.RWMol, conf: Chem.Conformer) -> int:
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
        if not all(0 < d <= _CARBOXYL_CO_MAX for d in co_dists):
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


def correct_carboxylate_bond_orders(mol: Chem.Mol) -> Chem.Mol:
    """Repair input bond orders that disagree with the 3D geometry.

    Re-sanitizes a corrected copy. Returns the input unchanged when no conformer
    or no qualifying finite, planar carboxylate geometry is available.
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
        logger.warning("geometry bond correction failed to sanitize", exc_info=True)
        return mol
    return out


def normalize_radical_oxygens(mol: Chem.Mol) -> Chem.Mol:
    """Restore formal charge on bare radical oxygens (DUD-style ``[O]``)."""
    rw = Chem.RWMol(mol)
    changed = False
    for atom in rw.GetAtoms():
        if (
            atom.GetSymbol() == "O"
            and atom.GetDegree() == 1
            and atom.GetTotalNumHs() == 0
            and atom.GetFormalCharge() == 0
            and atom.GetNumRadicalElectrons() > 0
        ):
            atom.SetFormalCharge(-1)
            atom.SetNumRadicalElectrons(0)
            changed = True
    if not changed:
        return mol
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        logger.warning(
            "Radical-oxygen normalization failed to sanitize mol", exc_info=True
        )
        return mol
    return out
