"""Graph isomorphism matching for ligand atom name mapping.

Matches heavy atoms between two RDKit representations of the same
molecule (e.g. a CIF-derived Mol and a SMILES-derived Mol) by molecular
graph isomorphism on the heavy-atom subgraph.
"""

from rdkit import Chem


def _heavy_atom_graph(
    mol: Chem.Mol,
) -> tuple[list[tuple[int, int]], dict[int, set[int]], dict[int, int]]:
    """Build a heavy-atom adjacency list with element labels.

    Returns:
        atoms: list of (rdkit_index, atomic_num) for heavy atoms
        adj: dict mapping heavy atom position -> set of neighbor positions
        idx_to_pos: dict mapping rdkit_index -> position in atoms list
    """
    atoms = []
    idx_to_pos = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = len(atoms)
        idx_to_pos[atom.GetIdx()] = pos
        atoms.append((atom.GetIdx(), atom.GetAtomicNum()))

    adj: dict[int, set[int]] = {i: set() for i in range(len(atoms))}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in idx_to_pos and b in idx_to_pos:
            adj[idx_to_pos[a]].add(idx_to_pos[b])
            adj[idx_to_pos[b]].add(idx_to_pos[a])

    return atoms, adj, idx_to_pos


def _vf2_match(
    atoms1: list[tuple[int, int]],
    adj1: dict[int, set[int]],
    atoms2: list[tuple[int, int]],
    adj2: dict[int, set[int]],
) -> dict[int, int] | None:
    """Simple VF2-style subgraph isomorphism for same-size graphs.

    Both graphs must have the same number of nodes. Returns a mapping
    {pos_in_graph1: pos_in_graph2} or None if no match found.
    """
    n = len(atoms1)
    if n != len(atoms2):
        return None

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}

    def is_feasible(p1: int, p2: int) -> bool:
        """Check whether assigning a node pair is graph-consistent.

        Args:
            p1: Candidate node index in graph 1.
            p2: Candidate node index in graph 2.

        Returns:
            ``True`` if the partial mapping remains feasible.
        """
        if atoms1[p1][1] != atoms2[p2][1]:
            return False
        for nbr1 in adj1[p1]:
            if nbr1 in mapping:
                if mapping[nbr1] not in adj2[p2]:
                    return False
        for nbr2 in adj2[p2]:
            if nbr2 in reverse:
                if reverse[nbr2] not in adj1[p1]:
                    return False
        return True

    def backtrack(depth: int) -> bool:
        """Depth-first search for a full node mapping.

        Args:
            depth: Current mapping depth.

        Returns:
            ``True`` if a complete isomorphism mapping is found.
        """
        if depth == n:
            return True

        p1 = depth
        elem = atoms1[p1][1]
        for p2 in range(n):
            if p2 in reverse:
                continue
            if atoms2[p2][1] != elem:
                continue
            if is_feasible(p1, p2):
                mapping[p1] = p2
                reverse[p2] = p1
                if backtrack(depth + 1):
                    return True
                del mapping[p1]
                del reverse[p2]
        return False

    if backtrack(0):
        return dict(mapping)
    return None


def match_heavy_atoms(
    pipeline_mol: Chem.Mol,
    cif_mol: Chem.Mol,
) -> dict[int, int]:
    """Match heavy atoms between pipeline and CIF Mol by graph isomorphism.

    Args:
        pipeline_mol: Mol from the SMILES pipeline (with H).
        cif_mol: Mol built from CIF coordinates (heavy atoms only).

    Returns:
        Dict mapping pipeline atom index -> CIF atom index for each
        heavy atom. Raises ValueError if no isomorphism found.
    """
    atoms1, adj1, idx_to_pos1 = _heavy_atom_graph(pipeline_mol)
    atoms2, adj2, idx_to_pos2 = _heavy_atom_graph(cif_mol)

    pos_mapping = _vf2_match(atoms1, adj1, atoms2, adj2)
    if pos_mapping is None:
        raise ValueError(
            f"Cannot match heavy atoms: pipeline has {len(atoms1)} "
            f"heavy atoms, CIF has {len(atoms2)}"
        )

    return {atoms1[p1][0]: atoms2[p2][0] for p1, p2 in pos_mapping.items()}
