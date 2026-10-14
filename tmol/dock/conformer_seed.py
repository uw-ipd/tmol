"""Low-energy ligand conformer libraries for docking initialization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from tmol.ligand import nonstandard_residue_info_from_mol2


DEFAULT_CONFORMER_RANDOM_SEED = 20260826
MAX_LIBRARY_SIZE = 50


@dataclass(frozen=True)
class ConformerLibrary:
    """Diverse MMFF94 conformers in ligand-block atom order."""

    coords: np.ndarray
    energies: np.ndarray
    atom_mapping: tuple[int, ...]
    atom_mapping_is_identity: bool


def _validate_parameters(
    n_conformers: int,
    energy_window: float,
    cluster_rmsd: float,
) -> None:
    if (
        not isinstance(n_conformers, int)
        or isinstance(n_conformers, bool)
        or n_conformers < 1
    ):
        raise ValueError("n_conformers must be a positive integer")
    if not math.isfinite(energy_window) or energy_window < 0:
        raise ValueError("energy_window must be finite and nonnegative")
    if not math.isfinite(cluster_rmsd) or cluster_rmsd < 0:
        raise ValueError("cluster_rmsd must be finite and nonnegative")


def _atom_type_elements(packed_block_types) -> dict[str, str]:
    return {
        atom_type.name: atom_type.element
        for atom_type in packed_block_types.chem_db.atom_types
    }


def _mapping_is_valid(
    mol: Chem.Mol,
    block_type,
    packed_block_types,
    mapping: tuple[int, ...],
) -> bool:
    """Check mapped atom elements and connectivity against the ligand block."""

    block_index = {atom.name: i for i, atom in enumerate(block_type.atoms)}
    atom_type_elements = _atom_type_elements(packed_block_types)
    elements_match = all(
        atom_type_elements[atom.atom_type]
        == mol.GetAtomWithIdx(mapping[index]).GetSymbol()
        for index, atom in enumerate(block_type.atoms)
    )
    bonds_match = all(
        mol.GetBondBetweenAtoms(mapping[block_index[a]], mapping[block_index[b]])
        is not None
        for a, b, *_ in block_type.bonds
    )
    return (
        elements_match
        and bonds_match
        and mol.GetNumBonds() == len(block_type.bonds)
    )


def _graph_atom_mapping(
    mol: Chem.Mol,
    block_type,
    packed_block_types,
) -> tuple[int, ...]:
    """Find an element/connectivity isomorphism from block atoms to mol atoms."""

    atom_type_elements = _atom_type_elements(packed_block_types)
    block_graph = nx.Graph()
    for index, atom in enumerate(block_type.atoms):
        block_graph.add_node(index, element=atom_type_elements[atom.atom_type])
    block_index = {atom.name: i for i, atom in enumerate(block_type.atoms)}
    for atom_a, atom_b, *_ in block_type.bonds:
        block_graph.add_edge(block_index[atom_a], block_index[atom_b])

    mol_graph = nx.Graph()
    for atom in mol.GetAtoms():
        mol_graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        mol_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        block_graph,
        mol_graph,
        node_match=lambda left, right: left["element"] == right["element"],
    )
    matches = matcher.isomorphisms_iter()
    try:
        match = next(matches)
    except StopIteration as exc:
        raise ValueError(
            "mol2 molecule and ligand block are not element/connectivity isomorphic"
        ) from exc
    try:
        next(matches)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "mol2-to-ligand graph mapping is ambiguous without matching atom names"
        )
    return tuple(match[index] for index in range(len(block_type.atoms)))


def _block_atom_mapping(
    mol: Chem.Mol,
    source_names: tuple[str, ...],
    block_type,
    packed_block_types,
) -> tuple[int, ...]:
    """Map ligand-block atom indices to mol2/RDKit atom indices."""

    block_names = tuple(atom.name for atom in block_type.atoms)
    if len(source_names) != len(set(source_names)):
        raise ValueError("mol2 atom names are not unique after disambiguation")
    if len(block_names) != len(set(block_names)):
        raise ValueError("ligand-block atom names are not unique")
    identity = tuple(range(len(block_names)))
    # Try, in order, the cheap correspondences (exact-name, name-set reorder,
    # preserved order), accepting the first that PRESERVES CONNECTIVITY. Names
    # can be normalized inconsistently between the block and the mol2 (block "C1"
    # need not be the same physical atom as mol2 "C1", e.g. hsp90), so a
    # name-based map that fails the connectivity check must fall back to a graph
    # isomorphism rather than raising.
    candidates: list[tuple[int, ...]] = []
    if block_names == source_names:
        candidates.append(identity)
    if set(block_names) == set(source_names):
        source_index = {name: index for index, name in enumerate(source_names)}
        candidates.append(tuple(source_index[name] for name in block_names))
    if len(block_names) == mol.GetNumAtoms():
        candidates.append(identity)
    for mapping in candidates:
        if _mapping_is_valid(mol, block_type, packed_block_types, mapping):
            return mapping
    mapping = _graph_atom_mapping(mol, block_type, packed_block_types)
    if not _mapping_is_valid(mol, block_type, packed_block_types, mapping):
        raise ValueError("mol2-to-ligand atom mapping does not preserve connectivity")
    return mapping


def _best_fit_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Kabsch-aligned RMSD for two same-order coordinate sets."""

    left_centered = left - np.mean(left, axis=0)
    right_centered = right - np.mean(right, axis=0)
    covariance = left_centered.T @ right_centered
    u, _, vt = np.linalg.svd(covariance)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    delta = left_centered @ rotation - right_centered
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _cluster_representatives(
    coords: np.ndarray,
    energies: np.ndarray,
    heavy_atoms: np.ndarray,
    cluster_rmsd: float,
) -> np.ndarray:
    distances = [
        _best_fit_rmsd(coords[i, heavy_atoms], coords[j, heavy_atoms])
        for i in range(1, coords.shape[0])
        for j in range(i)
    ]
    clusters = Butina.ClusterData(
        distances,
        coords.shape[0],
        cluster_rmsd,
        isDistData=True,
        reordering=True,
    )
    representatives = [
        min(cluster, key=lambda index: (energies[index], index))
        for cluster in clusters
    ]
    representatives.sort(key=lambda index: (energies[index], index))
    return np.asarray(representatives[:MAX_LIBRARY_SIZE], dtype=np.int64)


def generate_conformer_library(
    mol2_path: str | Path,
    block_type,
    packed_block_types,
    *,
    n_conformers: int = 200,
    energy_window: float = 20.0,
    cluster_rmsd: float = 0.5,
    random_seed: int = DEFAULT_CONFORMER_RANDOM_SEED,
) -> ConformerLibrary:
    """Embed, MMFF94-refine, filter, and cluster ligand conformers.

    ``n_conformers`` is the number of ETKDGv3 embedding attempts. The returned
    library is capped at :data:`MAX_LIBRARY_SIZE` conformers.
    """

    _validate_parameters(n_conformers, energy_window, cluster_rmsd)
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("random_seed must be an integer")

    path = Path(mol2_path)
    ligand_info = nonstandard_residue_info_from_mol2(path)
    mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not sanitize mol2 for ETKDG: {path}")
    Chem.AssignStereochemistryFrom3D(mol)
    source_names = tuple(ligand_info.atom_names)
    if mol.GetNumAtoms() != len(source_names):
        raise ValueError(
            "sanitized and established mol2 readers disagree on atom count"
        )
    mapping = _block_atom_mapping(
        mol, source_names, block_type, packed_block_types
    )
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("ligand is not fully parameterized by MMFF94")

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 1
    conformer_ids = tuple(
        AllChem.EmbedMultipleConfs(work, numConfs=n_conformers, params=params)
    )
    if not conformer_ids:
        raise ValueError("ETKDGv3 produced no ligand conformers")
    optimization = AllChem.MMFFOptimizeMoleculeConfs(
        work,
        numThreads=1,
        maxIters=400,
        mmffVariant="MMFF94",
    )
    statuses = np.asarray([entry[0] for entry in optimization], dtype=np.int32)
    energies = np.asarray([entry[1] for entry in optimization], dtype=np.float64)
    converged = (statuses == 0) & np.isfinite(energies)
    if not np.any(converged):
        raise ValueError("MMFF94 produced no converged conformers")
    minimum = float(np.min(energies[converged]))
    keep = converged & (energies <= minimum + energy_window)
    kept_ids = np.asarray(conformer_ids, dtype=np.int64)[keep]
    energies = energies[keep]
    if kept_ids.size == 0:
        raise ValueError("energy filtering removed every ligand conformer")

    coords = np.stack(
        [np.asarray(work.GetConformer(int(cid)).GetPositions()) for cid in kept_ids]
    )[:, mapping, :]
    heavy_atoms = np.asarray(
        [
            index
            for index, source_index in enumerate(mapping)
            if mol.GetAtomWithIdx(source_index).GetAtomicNum() > 1
        ],
        dtype=np.int64,
    )
    if heavy_atoms.size == 0:
        raise ValueError("ligand has no heavy atoms")
    representatives = _cluster_representatives(
        coords, energies, heavy_atoms, cluster_rmsd
    )
    return ConformerLibrary(
        coords=np.ascontiguousarray(coords[representatives], dtype=np.float64),
        energies=np.ascontiguousarray(energies[representatives], dtype=np.float64),
        atom_mapping=mapping,
        atom_mapping_is_identity=mapping == tuple(range(len(mapping))),
    )


__all__ = [
    "ConformerLibrary",
    "DEFAULT_CONFORMER_RANDOM_SEED",
    "MAX_LIBRARY_SIZE",
    "generate_conformer_library",
]
