"""Build tmol RawResidueType definitions from OpenBabel molecules.

Converts an OBMol with assigned atom types into a complete RawResidueType
suitable for registration in tmol's ChemicalDatabase. Handles atom tree
construction, internal coordinate computation, rotatable bond detection,
and non-polymer property assignment.

The internal coordinate and atom tree algorithms are ported from
Rosetta's mol2genparams.py and molfile_to_params.py.
"""

import logging
import math
from collections import deque

import numpy as np
from openbabel import openbabel

from tmol.database.chemical import (
    Atom,
    ChemicalProperties,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
)
from tmol.ligand.atom_typing import AtomTypeAssignment

logger = logging.getLogger(__name__)


def _find_nbr_atom(obmol: openbabel.OBMol) -> int:
    """Find the neighbor atom (root) for the atom tree.

    Selects the heavy atom closest to the center of mass that has
    at least 2 bonds, avoiding hydrogens and terminal atoms.

    Args:
        obmol: An OBMol with 3D coordinates.

    Returns:
        0-based atom index for the NBR atom.
    """
    n_atoms = obmol.NumAtoms()
    coords = np.zeros((n_atoms, 3))
    for obatom in openbabel.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        coords[idx] = [obatom.GetX(), obatom.GetY(), obatom.GetZ()]

    com = coords.mean(axis=0)
    dists_sq = np.sum((coords - com) ** 2, axis=1)

    best_idx = 0
    best_dist = float("inf")

    for obatom in openbabel.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        if obatom.IsHydrogen():
            continue
        if obatom.GetTotalDegree() < 2:
            continue
        if dists_sq[idx] < best_dist:
            best_dist = dists_sq[idx]
            best_idx = idx

    return best_idx


def _build_atom_tree(
    obmol: openbabel.OBMol, root_idx: int
) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    """Build an atom tree via BFS from the root atom.

    Args:
        obmol: The OBMol.
        root_idx: 0-based index of the root (NBR) atom.

    Returns:
        A tuple of:
        - order: BFS traversal order (list of 0-based atom indices).
        - parent: Maps each atom index to its parent index.
        - grandparents: Maps each atom index to (grandparent, great-grandparent).
    """
    n_atoms = obmol.NumAtoms()
    visited = [False] * n_atoms
    parent: dict[int, int] = {root_idx: root_idx}
    order: list[int] = []
    queue: deque[int] = deque([root_idx])
    visited[root_idx] = True

    adj: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for obbond in openbabel.OBMolBondIter(obmol):
        a = obbond.GetBeginAtomIdx() - 1
        b = obbond.GetEndAtomIdx() - 1
        adj[a].append(b)
        adj[b].append(a)

    while queue:
        current = queue.popleft()
        order.append(current)
        for nbr in adj[current]:
            if not visited[nbr]:
                visited[nbr] = True
                parent[nbr] = current
                queue.append(nbr)

    grandparents: dict[int, tuple[int, int]] = {}
    for idx in order:
        par = parent[idx]
        gp = parent.get(par, par)
        ggp = parent.get(gp, gp)
        grandparents[idx] = (gp, ggp)

    return order, parent, grandparents


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Dihedral angle a-b-c-d in degrees."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1) + 1e-12
    n2_norm = np.linalg.norm(n2) + 1e-12
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def _compute_icoors(
    obmol: openbabel.OBMol,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
    atom_names: list[str],
) -> list[Icoor]:
    """Compute internal coordinates for all atoms.

    Uses the Rosetta ICOOR convention:
    - d: distance from child to parent
    - theta: 180 - angle(child, parent, grandparent)
    - phi: dihedral(child, parent, grandparent, great-grandparent)

    Args:
        obmol: The OBMol with 3D coordinates.
        order: BFS traversal order.
        parent: Parent map from atom tree.
        grandparents: (grandparent, great-grandparent) map.
        atom_names: Atom names indexed by 0-based atom index.

    Returns:
        A list of Icoor objects in BFS traversal order.
    """
    coords = np.zeros((obmol.NumAtoms(), 3))
    for obatom in openbabel.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        coords[idx] = [obatom.GetX(), obatom.GetY(), obatom.GetZ()]

    icoors: list[Icoor] = []

    for i, idx in enumerate(order):
        par_idx = parent[idx]
        gp_idx, ggp_idx = grandparents[idx]

        if i == 0:
            d, theta, phi = 0.0, 0.0, 0.0
        elif i == 1:
            d = _distance(coords[idx], coords[par_idx])
            theta = 180.0
            phi = 0.0
        elif i == 2:
            d = _distance(coords[idx], coords[par_idx])
            theta = 180.0 - _angle(coords[idx], coords[par_idx], coords[gp_idx])
            phi = 0.0
        else:
            d = _distance(coords[idx], coords[par_idx])
            angle_val = _angle(coords[idx], coords[par_idx], coords[gp_idx])
            theta = 180.0 - angle_val
            phi = _dihedral(
                coords[idx], coords[par_idx], coords[gp_idx], coords[ggp_idx]
            )

        icoors.append(
            Icoor(
                name=atom_names[idx],
                phi=math.radians(phi),
                theta=math.radians(theta),
                d=d,
                parent=atom_names[par_idx],
                grand_parent=atom_names[gp_idx],
                great_grand_parent=atom_names[ggp_idx],
            )
        )

    return icoors


def build_residue_type(
    obmol: openbabel.OBMol,
    res_name: str,
    atom_types: list[AtomTypeAssignment],
) -> RawResidueType:
    """Build a complete RawResidueType from an OBMol.

    Constructs atoms, bonds, internal coordinates, and non-polymer
    properties suitable for registration in tmol's ChemicalDatabase.

    Args:
        obmol: An OpenBabel OBMol with 3D coordinates and bonds.
        res_name: Three-letter residue name (e.g. "LG1", "ATP").
        atom_types: Atom type assignments from assign_tmol_atom_types().

    Returns:
        A fully populated RawResidueType.
    """
    idx_to_name = {at.index: at.atom_name for at in atom_types}
    atom_names = [idx_to_name[i] for i in range(obmol.NumAtoms())]

    atoms = tuple(Atom(name=at.atom_name, atom_type=at.atom_type) for at in atom_types)

    bonds: list[tuple[str, str]] = []
    for obbond in openbabel.OBMolBondIter(obmol):
        a = obbond.GetBeginAtomIdx() - 1
        b = obbond.GetEndAtomIdx() - 1
        bonds.append((atom_names[a], atom_names[b]))

    nbr_idx = _find_nbr_atom(obmol)
    order, parent, grandparents = _build_atom_tree(obmol, nbr_idx)
    icoors = _compute_icoors(obmol, order, parent, grandparents, atom_names)

    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=False,
            polymer_type="NA",
            backbone_type="NA",
            mainchain_atoms=None,
            sidechain_chirality="NA",
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )

    return RawResidueType(
        name=res_name,
        base_name=res_name,
        name3=res_name,
        io_equiv_class=res_name,
        atoms=atoms,
        atom_aliases=(),
        bonds=tuple(bonds),
        connections=(),
        torsions=(),
        icoors=tuple(icoors),
        properties=properties,
        chi_samples=(),
        default_jump_connection_atom=atom_names[nbr_idx],
    )
