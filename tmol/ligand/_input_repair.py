"""Repairs for source files whose charges and bond orders are incomplete.

These encode an input-cleanup policy for the formats TMol reads, not a general
bond perception method, and must only be requested for a source that needs them.
Chemistry reaches AtomWorks already repaired, so nothing downstream has to guess
what a file meant.
"""

import logging

import biotite.structure as struc
import numpy as np
from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
from rdkit import Chem

logger = logging.getLogger(__name__)

# Below this, a C-O bond reads as double/delocalized rather than a hydroxyl
# single bond (carboxylate ~1.25 A, C=O ~1.21, C-OH ~1.31, diol C-O ~1.41).
_CARBOXYL_CO_MAX = 1.36
# Sum of the three bond angles at an sp2 (planar) carbon is 360; sp3 ~328.5.
_SP2_ANGLE_SUM_MIN = 355.0


# Valence a delocalized center fills, and the charge its singly-bonded,
# unprotonated terminal neighbours carry.
_DELOCALIZED_VALENCE = {"P": 5, "S": 6, "C": 4, "N": 4}
_DELOCALIZED_ANION = {"O": -1, "S": -1, "N": 0}
# Bonds a neutral atom of each element carries.
_NEUTRAL_VALENCE = {"N": 3, "O": 2, "P": 5, "S": 6, "C": 4}


def _delocalized_neighbors(mol, center, delocalized_bonds):
    """Neighbors sharing a delocalized bond with ``center`` only, ring or not (3GE7)."""
    neighbors = []
    for atom in center.GetNeighbors():
        pair = frozenset((center.GetIdx(), atom.GetIdx()))
        if pair not in delocalized_bonds:
            continue
        if any(
            frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            in delocalized_bonds
            and center.GetIdx() not in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in atom.GetBonds()
        ):
            return []
        neighbors.append(atom)
    return neighbors


def _localize(center, neighbors, n_double, conformer, charge, charges, synthesized):
    """Write ``n_double`` X=Y plus single bonds, charging the unprotonated remainder.

    Atoms in ``charges`` (declared) keep their charge; every charge written here
    is recorded in ``synthesized``, so a caller can compare the net it declared.
    """
    if conformer is not None:
        origin = np.asarray(conformer.GetAtomPosition(center.GetIdx()))
        neighbors = sorted(
            neighbors,
            key=lambda a: float(
                np.linalg.norm(
                    np.asarray(conformer.GetAtomPosition(a.GetIdx())) - origin
                )
            ),
        )
    center.SetIsAromatic(False)

    def protonated(atom):
        """Whether the file drew a hydrogen atom on this neighbour."""
        return any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())

    def terminal(atom):
        """Whether this neighbour hangs off the center by its only heavy bond."""
        return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() != 1) <= 1

    def demanded_charge(atom, order):
        """The charge the drawn bonds demand of this neighbour at that order."""
        neutral = _NEUTRAL_VALENCE.get(atom.GetSymbol())
        if neutral is None:
            return 0
        used = order + sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
            if bond.GetOtherAtomIdx(atom.GetIdx()) != center.GetIdx()
            and bond.GetBondType() != Chem.BondType.AROMATIC
        )
        return int(used - neutral)

    def wants_double(atom):
        """Whether a double bond fits this neighbour's declared charge or neutrality."""
        declared = charges.get(atom.GetIdx())
        if declared is not None:
            return declared == demanded_charge(atom, 2)
        return demanded_charge(atom, 2) <= 0

    # An uncharged guanidinium has no neutral option, so the geometry decides; a
    # declaration no arrangement fits is left unsanitizable for the fallback reader.
    undeclared = [atom for atom in neighbors if atom.GetIdx() not in charges]
    eligible = [a for a in neighbors if wants_double(a)] or undeclared or neighbors
    n_double = min(n_double, len(eligible))
    doubled = set(atom.GetIdx() for atom in eligible[:n_double])

    for atom in neighbors:
        bond = center.GetOwningMol().GetBondBetweenAtoms(center.GetIdx(), atom.GetIdx())
        bond.SetIsAromatic(False)
        atom.SetIsAromatic(False)
        double = atom.GetIdx() in doubled
        bond.SetBondType(Chem.BondType.DOUBLE if double else Chem.BondType.SINGLE)
        if atom.GetIdx() in charges:
            continue
        if protonated(atom):
            assigned = demanded_charge(atom, 2 if double else 1)
        elif double or not terminal(atom):
            assigned = 0
        else:
            assigned = charge
        atom.SetFormalCharge(assigned)
        synthesized[atom.GetIdx()] = assigned

    if center.GetIdx() not in charges:
        # the center holds what its rewritten bonds demand (a nitro N is +1)
        center.UpdatePropertyCache(strict=False)
        used = (
            sum(bond.GetBondTypeAsDouble() for bond in center.GetBonds())
            + center.GetTotalNumHs()
        )
        neutral = _NEUTRAL_VALENCE.get(center.GetSymbol())
        assigned = int(used - neutral) if neutral is not None else 0
        center.SetFormalCharge(assigned)
        if assigned:
            synthesized[center.GetIdx()] = assigned


def _infer_oxyacid_bonds(mol, charges, delocalized_bonds, synthesized=None):
    """Localize delocalized bonds: Tripos ``ar``, or orders geometry or valence refute.

    Such bonds joining a center to neighbours with no other one describe a
    delocalized group (carboxylate, phosphonate, sulfonate, nitro, amidine,
    guanidine), which has no Kekule structure as written. Each center gets its
    valence's double bonds, on the neighbours whose declared charge and hydrogens
    fit one, shortest first, and charges by valence.
    """
    synthesized = {} if synthesized is None else synthesized
    mol.UpdatePropertyCache(strict=False)
    for center in mol.GetAtoms():
        neighbors = _delocalized_neighbors(mol, center, delocalized_bonds)
        if len(neighbors) < 2 or len({a.GetSymbol() for a in neighbors}) != 1:
            continue
        anion = _DELOCALIZED_ANION.get(neighbors[0].GetSymbol())
        valence = _DELOCALIZED_VALENCE.get(center.GetSymbol())
        if anion is None or valence is None:
            continue
        spent = sum(
            bond.GetBondTypeAsDouble()
            for bond in center.GetBonds()
            if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            not in delocalized_bonds
        )
        n_double = int(valence - spent - len(neighbors))
        if not 0 < n_double <= len(neighbors):
            continue
        conformer = mol.GetConformer() if mol.GetNumConformers() else None
        _localize(center, neighbors, n_double, conformer, anion, charges, synthesized)
    mol.UpdatePropertyCache(strict=False)


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
    with short C-O bonds is a delocalized carboxylate, not a diol, whatever
    orders the input gives them (PDBbind v2020 2XEJ: a C.3 carboxyl with single
    C-O and C-OXT). Its bonds are localized as every delocalized group's are.
    """
    delocalized = set()
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
        delocalized.update(frozenset((c, o)) for o in term_os)
        logger.info("inferring COO- from geometry (carbon atom idx %d)", c)
    if delocalized:
        _infer_oxyacid_bonds(rw, {}, delocalized)
    return len(delocalized) // 2


def localize_overvalent_centers(mol: Chem.Mol) -> None:
    """Localize the double bonds to terminal atoms that put a center past its valence.

    CCD R6R writes its nitro group N(=O)=O; the group becomes N(=O)[O-] with N+.
    """
    delocalized = set()
    for atom in mol.GetAtoms():
        doubles = [
            b
            for b in atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(atom).GetDegree() == 1
        ]
        most = _DELOCALIZED_VALENCE.get(atom.GetSymbol())
        valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        if len(doubles) > 1 and most is not None and valence > most:
            delocalized.update(
                frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in doubles
            )
    if delocalized:
        _infer_oxyacid_bonds(mol, {}, delocalized)


def correct_carboxylate_bond_orders(mol: Chem.Mol) -> Chem.Mol:
    """Repair input bond orders that disagree with the 3D geometry.

    Re-sanitizes a corrected copy. Returns the input unchanged when no conformer
    or no qualifying finite, planar carboxylate geometry is available.
    """
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
    """Identify unobserved terminal O/N groups displaced at carbonyl/phosphoryl sites.

    A declared extra bond at a template carbonyl C or phosphoryl P replaces an absent
    single-bonded terminal O/N branch. Resolved atoms, carbonyl oxygens, and
    ambiguous alternatives are retained. Coordinates only establish whether
    atoms were observed; they do not determine chemical bond orders.
    ``connection_atoms`` contains template atom names with extra inter-residue bonds.
    """
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
