"""Repairs for source files whose charges and bond orders are incomplete.
An input-cleanup policy for the formats TMol reads, not general bond perception."""

import logging
from itertools import combinations

import biotite.structure as struc
import numpy as np
from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from rdkit import Chem

from tmol.ligand._icoor_tree import vertex_angle

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
    pos = conf.GetPositions()
    norms = np.linalg.norm(pos[neighbors] - pos[center], axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        return None
    return sum(
        np.degrees(vertex_angle(pos[i], pos[center], pos[j]))
        for i, j in combinations(neighbors, 2)
    )


def _infer_carboxylate_bonds(rw: Chem.RWMol, conf: Chem.Conformer) -> int:
    """Rewrite each planar carbon with two short terminal C-O bonds as ``C(=O)[O-]``.
    Returns the count; CIF SING/SING and mol2 ``ar`` carboxylates arrive as diols."""
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

        # the shorter bond is the carbonyl, whatever the input's atom order
        oa, ob = term_os if co_dists[0] <= co_dists[1] else term_os[::-1]
        for idx in (c, oa, ob):
            rw.GetAtomWithIdx(idx).SetIsAromatic(False)
        b_oa = rw.GetBondBetweenAtoms(c, oa)
        b_ob = rw.GetBondBetweenAtoms(c, ob)
        b_oa.SetIsAromatic(False)
        b_ob.SetIsAromatic(False)
        b_oa.SetBondType(Chem.BondType.DOUBLE)
        b_ob.SetBondType(Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(oa).SetFormalCharge(0)
        rw.GetAtomWithIdx(ob).SetFormalCharge(-1)
        n_fixed += 1
        logger.info("inferring COO- from geometry (carbon atom idx %d)", c)
    return n_fixed


def correct_carboxylate_bond_orders(mol: Chem.Mol) -> Chem.Mol:
    """A sanitized copy with geometry-repaired carboxylates, else ``mol``."""
    if mol.GetNumConformers() == 0:
        return mol
    rw = Chem.RWMol(mol)
    if _infer_carboxylate_bonds(rw, rw.GetConformer()) == 0:
        return mol
    return _sanitized(rw, mol, "geometry bond correction failed to sanitize")


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
    return _sanitized(rw, mol, "Radical-oxygen normalization failed to sanitize mol")


def _sanitized(rw: Chem.RWMol, mol: Chem.Mol, failure: str) -> Chem.Mol:
    """The sanitized edit ``rw``, or the unedited ``mol`` (logging ``failure``)."""
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        logger.warning(failure, exc_info=True)
        return mol
    return out


def get_absent_substitution_leaving_groups(
    template: struc.AtomArray, residue: struc.AtomArray, connection_atoms: set[str]
) -> dict[str, tuple[str, ...]]:
    """{site: unobserved terminal O/N branch} an extra bond at a template carbonyl C or
    phosphoryl P displaces; resolved atoms and ambiguous alternatives are kept."""
    if template.bonds is None or not connection_atoms:
        return {}
    observed = set(residue.atom_name[np.isfinite(residue.coord).all(axis=-1)])
    result = {}
    for index in np.flatnonzero(
        np.isin(template.element, ("C", "P"))
        & np.isin(template.atom_name, list(connection_atoms))
    ):
        neighbors, orders = template.bonds.get_bonds(index)
        if not np.any(
            (template.element[neighbors] == "O")
            & (orders == int(struc.BondType.DOUBLE))
        ):
            continue
        candidates = []
        for neighbor, order in zip(neighbors, orders, strict=True):
            if order != int(struc.BondType.SINGLE) or template.element[
                neighbor
            ] not in ("N", "O"):
                continue
            branch, _ = template.bonds.get_bonds(neighbor)
            if any(
                i != index and template.element[i] not in HYDROGEN_LIKE_SYMBOLS
                for i in branch
            ):
                continue
            group = tuple(
                str(template.atom_name[i])
                for i in (neighbor, *(i for i in branch if i != index))
            )
            if observed.isdisjoint(group):
                candidates.append(group)
        if len(candidates) == 1:
            result[str(template.atom_name[index])] = candidates[0]
    return result
