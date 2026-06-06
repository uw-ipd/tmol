"""Rosetta SetupTopology ring/hybridization state for atom typing.

Ports the pre-classifier state that Rosetta's mol2genparams constructs from
mol2 Tripos atom-type columns and bond tables before calling
``AtomTypeClassifier``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from rdkit import Chem

# Rosetta Types.SPECIAL_HYBRIDS (subset used by PLI mol2 inputs)
_SPECIAL_HYBRIDS: dict[str, int] = {
    "N.pl3": 2,
    "N.am": 8,
    "N.aro": 2,
    "C.cat": 2,
    "O.co2": 2,
    "S.o": 3,
    "S.o2": 5,
    "N.4": 3,
    "c": 2,
    "ca": 9,
    "c1": 1,
    "c2": 2,
    "c3": 3,
    "cc": 2,
    "cd": 2,
    "ce": 2,
    "cf": 2,
    "ch": 1,
    "cg": 1,
    "cp": 2,
    "cx": 2,
    "cy": 3,
    "cz": 2,
    "o": 2,
    "os": 3,
    "oh": 3,
    "n": 2,
    "na": 9,
    "nb": 2,
    "no": 2,
    "nc": 2,
    "nd": 2,
    "ne": 2,
    "nh": 0,
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "n4": 3,
    "s": 3,
    "ss": 3,
    "sh": 3,
    "s4": 2,
    "s6": 5,
    "sy": 5,
}

_ATOMIC_NUM_TO_ELEMENT = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    15: "P",
    16: "S",
}


class RosettaRingState(NamedTuple):
    rings: list[tuple[int, ...]]
    atms_aro: set[int]
    atms_strained: set[int]


def hyb_from_tripos_subtype(sybyl: str) -> int:
    """Map a Tripos/GAFF atom-type string to Rosetta ``hyb`` integer codes."""
    s = (sybyl or "").strip()
    if not s:
        return 0
    key = s if s in _SPECIAL_HYBRIDS else s.lower()
    if key in _SPECIAL_HYBRIDS:
        return _SPECIAL_HYBRIDS[key]
    if "." in s:
        suffix = s.split(".")[-1].lower()
        if suffix in ("1", "2", "3"):
            return int(suffix)
        if suffix == "ar":
            return 9
    return 0


def _bond_order_code(bond: Chem.Bond) -> float:
    if bond.GetIsAromatic():
        return 9.0
    return float(bond.GetBondTypeAsDouble())


def _get_path(
    bond_tree: np.ndarray, start: int, end: int, alt_cut: tuple[int, int] | None = None
) -> list[int]:
    """Shortest path between ``start`` and ``end`` in an unweighted graph."""
    visited = {start}
    queue: list[tuple[int, list[int]]] = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        for nbr in np.nonzero(bond_tree[node])[0]:
            ni = int(nbr)
            if alt_cut is not None and {node, ni} == set(alt_cut):
                continue
            if ni in visited:
                continue
            new_path = path + [ni]
            if ni == end:
                return new_path
            visited.add(ni)
            queue.append((ni, new_path))
    return []


def _ring_duplicate(
    rings: list[tuple[int, ...]], candidate: list[int]
) -> tuple[int, ...] | None:
    cand_set = set(candidate)
    for ring in rings:
        if set(ring) == cand_set:
            return ring
    return None


def _classify_ring_type(
    mol: Chem.Mol,
    ring: tuple[int, ...],
    hyb_by_idx: dict[int, int],
    coords: np.ndarray | None,
) -> tuple[bool, bool]:
    """Return ``(is_aromatic, is_strained)`` for a ring atom index tuple."""
    ring_size = len(ring)
    is_aro = True

    if ring_size > 6:
        if coords is None:
            is_aro = False
        else:
            ring_xyz = coords[list(ring)]
            ring_xyz = ring_xyz - np.mean(ring_xyz, axis=0)
            cov = np.cov(ring_xyz, rowvar=False)
            evals = np.linalg.eigvalsh(cov)
            if float(np.min(evals)) > 1e-2:
                is_aro = False
    else:
        nsp2_n = 0
        nsp3_os = 0
        nsp3 = 0
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            hyb = hyb_by_idx.get(idx, 0)
            z = atom.GetAtomicNum()
            elem = _ATOMIC_NUM_TO_ELEMENT.get(z, "")
            if hyb == 3:
                nsp3 += 1
                for bond in atom.GetBonds():
                    if _bond_order_code(bond) not in (2.0, 4.0, 9.0):
                        is_aro = False
                if elem in ("O", "S"):
                    nsp3_os += 1
            elif hyb == 2 and elem == "N":
                nsp2_n += 1
            if ring_size == 5:
                is_aro = (nsp3 == nsp3_os) or (nsp2_n == 2)

    is_strained = ring_size <= 4
    return is_aro, is_strained


def detect_rosetta_rings(
    mol: Chem.Mol,
    hyb_by_idx: dict[int, int],
    *,
    coords: np.ndarray | None = None,
) -> RosettaRingState:
    """Detect rings and aromatic/strained atom sets Rosetta-style."""
    n = mol.GetNumAtoms()
    bond_tree = np.zeros((n, n), dtype=np.int8)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bond_tree[a, b] = 1
        bond_tree[b, a] = 1

    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
    except ImportError:
        rings = [tuple(int(i) for i in ring) for ring in Chem.GetSSSR(mol)]
        atms_aro: set[int] = set()
        atms_strained: set[int] = set()
        for ring in rings:
            is_aro, is_strained = _classify_ring_type(mol, ring, hyb_by_idx, coords)
            if is_strained:
                atms_strained.update(ring)
            if is_aro:
                atms_aro.update(ring)
        return RosettaRingState(
            rings=rings, atms_aro=atms_aro, atms_strained=atms_strained
        )

    tree = minimum_spanning_tree(bond_tree)
    tree_symm = np.maximum(tree.toarray(), tree.T.toarray())
    cycle_edges = np.transpose(np.nonzero(np.triu(bond_tree != tree_symm)))

    rings: list[tuple[int, ...]] = []
    for i, j in cycle_edges:
        ring_i = _get_path(bond_tree, int(i), int(j))
        if not ring_i:
            continue
        dupl = _ring_duplicate(rings, ring_i)
        if dupl is not None:
            alt = _get_path(bond_tree, int(i), int(j), alt_cut=(dupl[0], dupl[-1]))
            if alt and _ring_duplicate(rings, alt) is None:
                rings.append(tuple(alt))
        else:
            rings.append(tuple(ring_i))
            alt = _get_path(bond_tree, int(i), int(j), alt_cut=(ring_i[0], ring_i[1]))
            if alt and _ring_duplicate(rings, alt) is None:
                rings.append(tuple(alt))

    atms_aro: set[int] = set()
    atms_strained: set[int] = set()
    for ring in rings:
        is_aro, is_strained = _classify_ring_type(mol, ring, hyb_by_idx, coords)
        if is_strained:
            atms_strained.update(ring)
        if is_aro:
            atms_aro.update(ring)

    return RosettaRingState(rings=rings, atms_aro=atms_aro, atms_strained=atms_strained)


def build_hyb_by_idx_from_subtypes(
    mol: Chem.Mol, source_subtype_by_idx: dict[int, str]
) -> dict[int, int]:
    """Assign Rosetta ``hyb`` codes from Tripos subtype tags."""
    from tmol.ligand.atom_typing import (
        _assign_missing_hybridization,
        _hyb_from_atom_and_subtype,
    )

    hyb_by_idx: dict[int, int] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sub = source_subtype_by_idx.get(idx, "").strip()
        if not sub or sub == "?":
            hyb_by_idx[idx] = 0
        elif "." in sub:
            hyb_by_idx[idx] = hyb_from_tripos_subtype(sub)
        else:
            hyb_by_idx[idx] = _hyb_from_atom_and_subtype(atom, sub)

    atms_aro: set[int] = set()
    _assign_missing_hybridization(mol, hyb_by_idx, atms_aro)
    return hyb_by_idx
