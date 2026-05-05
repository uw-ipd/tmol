"""Build tmol RawResidueType definitions from RDKit molecules.

Converts a Chem.Mol with assigned atom types into a complete RawResidueType
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
from rdkit import Chem

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


def _mol_coords(mol: Chem.Mol) -> np.ndarray:
    """Return an (N, 3) float array of 3D coordinates from mol's first conformer.

    If the mol has no conformer (e.g. a SMILES-only unit-test input),
    return zeros — the caller's icoor geometry will be degenerate but the
    atom tree and topology outputs still build correctly.
    """
    n = mol.GetNumAtoms()
    if mol.GetNumConformers() == 0:
        return np.zeros((n, 3))
    conf = mol.GetConformer(0)
    coords = np.zeros((n, 3))
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = (p.x, p.y, p.z)
    return coords


def _find_nbr_atom(
    mol: Chem.Mol, coords: np.ndarray, skip_indices: set[int] | None = None
) -> int:
    """Find the neighbor atom (root) for the atom tree.

    Selects the heavy atom closest to the center of mass that has
    at least 2 bonds, avoiding hydrogens, terminal atoms, and any
    indices in ``skip_indices``.

    Args:
        mol: A Chem.Mol.
        coords: Pre-computed (N, 3) coordinate array from ``_mol_coords``.
        skip_indices: Optional set of 0-based atom indices to exclude.

    Returns:
        0-based atom index for the NBR atom.

    Raises:
        ValueError: If no valid root atom can be found.
    """
    if skip_indices is None:
        skip_indices = set()

    com = coords.mean(axis=0)
    dists_sq = np.sum((coords - com) ** 2, axis=1)

    best_idx = -1
    best_dist = float("inf")

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in skip_indices:
            continue
        if atom.GetAtomicNum() == 1:
            continue
        if atom.GetDegree() < 2:
            continue
        if dists_sq[idx] < best_dist:
            best_dist = dists_sq[idx]
            best_idx = idx

    if best_idx < 0:
        # Fallback: pick the first heavy atom with any bonds
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in skip_indices:
                continue
            if atom.GetAtomicNum() == 1:
                continue
            best_idx = idx
            break

    if best_idx < 0:
        raise ValueError("No valid root atom found (mol has no heavy atoms)")

    return best_idx


def _build_atom_tree(
    mol: Chem.Mol, root_idx: int
) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    """Build an atom tree via simple BFS from the root.

    Returns:
        A tuple of:
        - order: BFS traversal order (parents always precede children).
        - parent: Maps each atom index to its parent index (root maps to itself).
        - grandparents: Maps each atom index to (grandparent, great-grandparent).
    """
    n_atoms = mol.GetNumAtoms()
    visited = [False] * n_atoms
    parent: dict[int, int] = {root_idx: root_idx}
    order: list[int] = []
    queue: deque[int] = deque([root_idx])
    visited[root_idx] = True

    adj: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)

    while queue:
        current = queue.popleft()
        order.append(current)
        for nbr in adj[current]:
            if visited[nbr]:
                continue
            visited[nbr] = True
            parent[nbr] = current
            queue.append(nbr)

    bfs_position = {idx: i for i, idx in enumerate(order)}

    def _is_heavy(i: int) -> bool:
        return mol.GetAtomWithIdx(i).GetAtomicNum() != 1

    def _pick_neighbor(
        of: int, exclude: set[int], heavy_only: bool = False
    ) -> int | None:
        candidates = [n for n in adj[of] if n not in exclude]
        if heavy_only:
            candidates = [n for n in candidates if _is_heavy(n)]
        if not candidates:
            return None
        candidates.sort(key=lambda n: bfs_position.get(n, len(order)))
        return candidates[0]

    grandparents: dict[int, tuple[int, int]] = {}
    for idx in order:
        idx_heavy = _is_heavy(idx)
        par = parent[idx]
        gp = parent.get(par, par)
        if gp == par and idx != root_idx:
            sub = _pick_neighbor(par, exclude={idx, par}, heavy_only=idx_heavy)
            if sub is not None:
                gp = sub
        ggp = parent.get(gp, gp)
        if ggp == gp and idx != root_idx:
            sub = _pick_neighbor(par, exclude={idx, par, gp}, heavy_only=idx_heavy)
            if sub is None:
                sub = _pick_neighbor(gp, exclude={par, gp, idx}, heavy_only=idx_heavy)
            if sub is not None:
                ggp = sub
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
    mol: Chem.Mol,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
    atom_names: list[str],
    coords: np.ndarray | None = None,
) -> list[Icoor]:
    """Compute internal coordinates for all atoms.

    Uses the Rosetta ICOOR convention:
    - d: distance from child to parent
    - theta: 180 - angle(child, parent, grandparent)
    - phi: dihedral(child, parent, grandparent, great-grandparent)

    Args:
        mol: The Chem.Mol with 3D coordinates.
        order: BFS traversal order.
        parent: Parent map from atom tree.
        grandparents: (grandparent, great-grandparent) map.
        atom_names: Atom names indexed by 0-based atom index.
        coords: Pre-computed (N, 3) array. Computed from mol if None.

    Returns:
        A list of Icoor objects in BFS traversal order.
    """
    if coords is None:
        coords = _mol_coords(mol)

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


_AMIDE_N_TYPES = {"Nad", "Nad3"}
_GUANIDINIUM_N_TYPES = {"Ngu1", "Ngu2"}
_PLANAR_N_TYPES = _AMIDE_N_TYPES | _GUANIDINIUM_N_TYPES

# Rosetta convention: rings larger than this are treated as
# non-cyclic for the purposes of the bond's is_in_ring flag.
_MAX_RING_SIZE_TREATED_AS_RING = 8


def _is_in_large_ring(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """True if every cycle this bond participates in is >8 atoms."""
    if not bond.IsInRing():
        return False
    ri = mol.GetRingInfo()
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    for ring in ri.AtomRings():
        if a in ring and b in ring and len(ring) <= _MAX_RING_SIZE_TREATED_AS_RING:
            return False
    return True


def _is_planar_resonance_bond(
    atom_a: Chem.Atom, atom_b: Chem.Atom, type_a: str, type_b: str
) -> bool:
    """True iff this N-C bond is the planar partner in an acyclic
    resonance group — i.e. the amide N-C(=O) or the guanidinium N-C(N)(N)
    bond, but NOT other N-C connections incident to the same N.

    Reuses the atom-type classifier's existing detection: ``Nad`` /
    ``Nad3`` mark amide nitrogens and ``Ngu1`` / ``Ngu2`` mark
    guanidinium nitrogens. We only upgrade the specific N-C bond whose
    C is the resonance partner (a carbonyl C for amide; a multi-N C for
    guanidinium); methylene tethers etc. stay SINGLE.
    """
    if type_a in _PLANAR_N_TYPES and atom_b.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_a, atom_b, type_a
    elif type_b in _PLANAR_N_TYPES and atom_a.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_b, atom_a, type_b
    else:
        return False

    other_neighbors = (
        nbr for nbr in c_atom.GetNeighbors() if nbr.GetIdx() != n_atom.GetIdx()
    )
    if n_type in _AMIDE_N_TYPES:
        # C must have a C=O carbonyl on its other side.
        return any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(c_atom).GetAtomicNum() == 8
            for b in c_atom.GetBonds()
            if b.GetOtherAtom(c_atom).GetIdx() != n_atom.GetIdx()
        )
    # guanidinium: C must have ≥2 other N neighbors.
    return sum(1 for nbr in other_neighbors if nbr.GetAtomicNum() == 7) >= 2


def build_residue_type(
    mol: Chem.Mol,
    res_name: str,
    atom_types: list[AtomTypeAssignment],
    atom_aliases: tuple = (),
) -> RawResidueType:
    """Build a complete RawResidueType from a Chem.Mol.

    Constructs atoms, bonds, internal coordinates, and non-polymer
    properties suitable for registration in tmol's ChemicalDatabase.

    Atoms with unknown elements (atomic number 0, e.g. metals that
    lost identity during SMILES roundtrip) are silently dropped.

    Args:
        mol: An RDKit Mol with 3D coordinates and bonds.
        res_name: Three-letter residue name (e.g. "LG1", "ATP").
        atom_types: Atom type assignments from assign_tmol_atom_types().
        atom_aliases: Optional tuple of AtomAlias for CIF name mapping.

    Returns:
        A fully populated RawResidueType.
    """
    dropped = [at for at in atom_types if at.element == "*"]
    if dropped:
        logger.warning(
            "%s: dropping %d atom(s) with unknown element: %s",
            res_name,
            len(dropped),
            ", ".join(at.atom_name for at in dropped),
        )
        keep_indices = {at.index for at in atom_types if at.element != "*"}
        atom_types = [at for at in atom_types if at.element != "*"]
    else:
        keep_indices = None

    idx_to_name = {at.index: at.atom_name for at in atom_types}
    atom_names = [idx_to_name.get(i) for i in range(mol.GetNumAtoms())]
    atom_type_by_name = {at.atom_name: at.atom_type for at in atom_types}

    atoms = tuple(Atom(name=at.atom_name, atom_type=at.atom_type) for at in atom_types)

    bonds: list[tuple[str, str, str]] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if atom_names[a] is None or atom_names[b] is None:
            continue

        # Kekulé first — Frank's reference .tmol files emit DOUBLE/SINGLE
        # for ring bonds when they carry a definite bond order, and only
        # fall back to AROMATIC when the bond was never kekulized (e.g.
        # acyclic resonance groups like amide N-C(=O)). Reading the bond
        # order directly catches the kekulized case; the GetIsAromatic
        # branch only fires when the bond type is still AROMATIC at this
        # point, i.e. kekulization was skipped or failed.
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            b_type = "SINGLE"
        elif bt == Chem.BondType.DOUBLE:
            b_type = "DOUBLE"
        elif bt == Chem.BondType.TRIPLE:
            b_type = "TRIPLE"
        elif bt == Chem.BondType.AROMATIC or bond.GetIsAromatic():
            b_type = "AROMATIC"
        else:
            b_type = "SINGLE"

        # Planar acyclic resonance (amide / guanidinium) — RDKit reports
        # the bond as SINGLE because it isn't on a Hückel ring, but the
        # geometry is planar and the downstream scoring treats this as
        # an aromatic bond.
        if b_type == "SINGLE" and _is_planar_resonance_bond(
            mol.GetAtomWithIdx(a),
            mol.GetAtomWithIdx(b),
            atom_type_by_name.get(atom_names[a], ""),
            atom_type_by_name.get(atom_names[b], ""),
        ):
            b_type = "AROMATIC"

        # 4th field: is_in_ring (for scoring; rings >8 atoms count as
        # not-in-a-ring, matching Rosetta convention).
        is_in_ring = bond.IsInRing() and not _is_in_large_ring(mol, bond)
        bonds.append((atom_names[a], atom_names[b], b_type, is_in_ring))

    coords = _mol_coords(mol)
    dropped_indices = (
        (set(range(mol.GetNumAtoms())) - keep_indices) if keep_indices else None
    )
    nbr_idx = _find_nbr_atom(mol, coords, skip_indices=dropped_indices)
    order, parent, grandparents = _build_atom_tree(mol, nbr_idx)
    if keep_indices is not None:
        order = [i for i in order if i in keep_indices]

    valid_set = set(order)
    for idx in order:
        if parent[idx] not in valid_set:
            parent[idx] = idx
        gp, ggp = grandparents[idx]
        if gp not in valid_set:
            gp = parent[idx]
        if ggp not in valid_set:
            ggp = gp
        grandparents[idx] = (gp, ggp)

    icoors = _compute_icoors(mol, order, parent, grandparents, atom_names, coords)

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
        atom_aliases=atom_aliases,
        bonds=tuple(bonds),
        connections=(),
        # NOTE: Ligand torsions are intentionally empty
        torsions=(),
        icoors=tuple(icoors),
        properties=properties,
        chi_samples=(),
        default_jump_connection_atom=atom_names[nbr_idx],
    )
