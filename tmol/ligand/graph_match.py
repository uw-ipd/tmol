"""Graph isomorphism matching for ligand atom name mapping.

Matches heavy atoms between two RDKit representations of the same
molecule (e.g. a CIF-derived Mol and a SMILES-derived Mol) by molecular
graph isomorphism on the heavy-atom subgraph.
"""

import networkx as nx
import networkx.algorithms.isomorphism as iso
from rdkit import Chem


def _heavy_atom_graph(mol: Chem.Mol) -> nx.Graph:
    """Heavy-atom graph with an element label per node (rdkit index -> element).

    Bond orders are ignored: the two Mols may perceive bonds differently
    (Kekule vs aromatic, tautomers), so matching is on element + connectivity.
    """
    g = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            g.add_node(atom.GetIdx(), z=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() != 1 and b.GetAtomicNum() != 1:
            g.add_edge(a.GetIdx(), b.GetIdx())
    return g


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
    g1 = _heavy_atom_graph(pipeline_mol)
    g2 = _heavy_atom_graph(cif_mol)
    gm = iso.GraphMatcher(g1, g2, node_match=iso.categorical_node_match("z", None))
    if not gm.is_isomorphic():
        raise ValueError(
            f"Cannot match heavy atoms: pipeline has {g1.number_of_nodes()} "
            f"heavy atoms, CIF has {g2.number_of_nodes()}"
        )
    return dict(gm.mapping)
