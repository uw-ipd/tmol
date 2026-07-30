"""Corrections to the generated (OpenBabel) 3D structure.

In some cases, openbabel-generated conformers are incorrect in known,
predicible ways.  This module handles those predictions, correcting
very specific geometry issues before generating parameters.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

# Bond orders that make an atom sp2, and hence conjugate its substituents.
_SP2_BONDS = (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC)

# Ideal sp2 substituent angle: each N-H sits 120 deg off the N->heavy bond,
# i.e. 60 deg either side of the bisector pointing away from that neighbor.
_SP2_HALF_ANGLE = math.radians(60.0)


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


def _is_sp2(atom) -> bool:
    """True if the atom carries a double/aromatic bond (resonance -> sp2)."""
    return any(b.GetBondType() in _SP2_BONDS for b in atom.GetBonds())


def planarize_conjugated_nh2(mol: Chem.Mol) -> list[str]:
    """Make -NH2 groups conjugated to an sp2 center planar.

    Correct proton geomoetry around amine N bound to sp2 heavy atoms.
    Both hydrogens are rebuilt in the neighbor's plane, 120 deg off the C-N
    bond, preserving each N-H bond length and each hydrogen's original side so
    the correction is minimal.
    """
    conf = mol.GetConformer()
    fixed: list[str] = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        hs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(hs) != 2 or len(heavy) != 1:
            continue
        nbr = heavy[0]
        if not _is_sp2(nbr):
            continue  # sp3 neighbor: a genuine pyramidal amine, leave it alone
        # the sp2 neighbor's other heavy atoms define the conjugation plane
        refs = [
            n
            for n in nbr.GetNeighbors()
            if n.GetIdx() != atom.GetIdx() and n.GetAtomicNum() != 1
        ]
        if len(refs) < 2:
            continue

        pos = lambda i: np.array(conf.GetAtomPosition(i), dtype=float)  # noqa: E731
        n_xyz = pos(atom.GetIdx())
        a_xyz = pos(nbr.GetIdx())
        normal = _unit(
            np.cross(pos(refs[0].GetIdx()) - a_xyz, pos(refs[1].GetIdx()) - a_xyz)
        )
        if normal is None:
            continue  # degenerate/collinear reference atoms
        bisector = _unit(n_xyz - a_xyz)  # in plane, pointing away from nbr
        if bisector is None:
            continue
        in_plane = _unit(np.cross(normal, bisector))  # in plane, perp to C-N
        if in_plane is None:
            continue

        # keep each H on the side it already occupies; if the generated geometry
        # is degenerate and both land on one side, split them.
        offsets = [pos(h.GetIdx()) - n_xyz for h in hs]
        sides = [1.0 if float(np.dot(v, in_plane)) >= 0.0 else -1.0 for v in offsets]
        if sides[0] == sides[1]:
            sides = [1.0, -1.0]

        before = _angle_sum_deg(n_xyz, a_xyz, [n_xyz + v for v in offsets])
        for h, v, side in zip(hs, offsets, sides):
            d = float(np.linalg.norm(v))
            new = n_xyz + d * (
                math.cos(_SP2_HALF_ANGLE) * bisector
                + side * math.sin(_SP2_HALF_ANGLE) * in_plane
            )
            conf.SetAtomPosition(h.GetIdx(), new.tolist())

        if before is not None and before < 355.0:
            fixed.append(
                f"planarized conjugated NH2 on atom {atom.GetIdx()} "
                f"(angle sum {before:.1f} -> 360.0 deg)"
            )
    return fixed


def _angle_sum_deg(center, a, hs):
    """Sum of the three bond angles about a 3-substituent center, in degrees."""
    vs = [_unit(np.asarray(x) - center) for x in [a] + list(hs)]
    if any(v is None for v in vs):
        return None
    total = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            total += math.degrees(
                math.acos(max(-1.0, min(1.0, float(np.dot(vs[i], vs[j])))))
            )
    return total


# Corrections applied, in order, to every generated structure.
_CORRECTIONS = (planarize_conjugated_nh2,)


def correct_generated_geometry(mol: Chem.Mol) -> list[str]:
    """Repair known defects in the generated conformer, in place.

    Apply all corrections specified in _CORRECTIONS.
    """
    if mol.GetNumConformers() == 0:
        return []
    applied: list[str] = []
    for correction in _CORRECTIONS:
        try:
            applied.extend(correction(mol))
        except Exception as err:  # noqa: BLE001  a correction must never break prep
            logger.warning(
                "geometry correction %s failed: %s", correction.__name__, err
            )
    return applied
