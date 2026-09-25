"""Repairs for source files whose charges and bond orders are incomplete.

These encode an input-cleanup policy for the formats TMol reads, not a general
bond perception method, and must only be requested for a source that needs them.
Chemistry reaches AtomWorks already repaired, so nothing downstream has to guess
what a file meant.
"""

import functools
import logging
from collections import defaultdict, deque

import biotite.structure as struc
import networkx as nx
import numpy as np
from atomworks.constants import HYDROGEN_LIKE_SYMBOLS
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
        # A carbonyl already written is a carboxylic acid's, not a carboxylate
        #    recorded as a geminal diol: an acid's oxygens are both terminal and
        #    both its C-O bonds are short enough to reach here, so only a centre
        #    with no double bond at all is missing one.
        if any(
            rw.GetBondBetweenAtoms(c, o).GetBondType() == Chem.BondType.DOUBLE
            for o in term_os
        ):
            continue
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        angle_sum = _sp2_angle_sum(conf, c, nbrs)
        if angle_sum is None or angle_sum < _SP2_ANGLE_SUM_MIN:
            continue

        # The carbonyl is the shorter bond; taking them in neighbour order would
        #    let the input's atom ordering decide which oxygen carries the charge.
        oa, ob = term_os if co_dists[0] <= co_dists[1] else term_os[::-1]
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


# ---------------------------------------------------------------------------
# Leaving groups
#
# A component's leaving atoms are the ones displaced when it forms a bond. The
# grouping is a property of the template's bond graph and its leaving-atom
# flags, so it lives here with the rest of the chemistry TMol resolves for
# itself rather than asking AtomWorks for.
# ---------------------------------------------------------------------------


def _find_connected_components_after_removal(
    graph: nx.Graph, node_to_remove: int
) -> list[list[int]]:
    """Identifies connected components that would form after removing a node from a graph.

    Args:
        graph: The input graph.
        node_to_remove: The node to hypothetically remove.

    Returns:
        List of lists containing node indices in each new component.
    """
    # Only neighbors of the removed atom seed traversals; disconnected
    # components remain excluded without constructing temporary graphs.
    unvisited = set(graph.neighbors(node_to_remove))
    components = []
    while unvisited:
        start = unvisited.pop()
        seen = {node_to_remove, start}
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
        unvisited.difference_update(component)
    return components


@functools.lru_cache(maxsize=128)
def _leaving_atom_groups(
    atom_name: tuple[str, ...],
    element: tuple[str, ...],
    leaving: tuple[bool, ...],
    bonds: tuple[tuple[int, int], ...],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Cache immutable chemical topology, without retaining arrays or coordinates."""
    leaving_atom_names = defaultdict(list)
    is_leaving_atom = np.asarray(leaving, dtype=bool)

    # ... compute the leaving groups based on the bond graph and annotation
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(range(len(atom_name)))
    bond_graph.add_edges_from(bonds)
    for atom_idx in range(len(atom_name)):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(
            bond_graph, atom_idx
        )

        # ... check if all atoms in the connected group are flagged as leaving atoms
        #     by the CCD entry
        for connected_group in connected_groups:
            heavy_atoms: list[int] = list(
                filter(lambda x: element[x] != "H", connected_group)
            )
            is_leaving_group = (
                all(is_leaving_atom[heavy_atoms])
                if len(heavy_atoms) > 0
                else all(is_leaving_atom[connected_group])
            )

            if is_leaving_group:
                leaving_atom_names[atom_name[atom_idx]].append(
                    tuple(atom_name[idx] for idx in connected_group)
                )

    # ... turn leaving_atom_names into a dictionary of tuples
    leaving_atom_names = {k: tuple(v) for k, v in leaving_atom_names.items()}

    return leaving_atom_names


def get_leaving_atom_groups(
    chem_comp: struc.AtomArray,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Find detachable groups using a template's bonds and leaving-atom flags.

    Keys name the attachment atom; each value contains its separate leaving
    groups. Unannotated templates declare no leaving groups. The input is not
    modified, and its residue code is never used for dictionary lookup.
    """
    if "is_leaving_atom" not in chem_comp.get_annotation_categories() or not np.any(
        chem_comp.is_leaving_atom
    ):
        return {}
    if chem_comp.bonds is None:
        raise ValueError("Leaving-group detection requires template bonds.")
    return dict(
        _leaving_atom_groups(
            tuple(chem_comp.atom_name),
            tuple(chem_comp.element),
            tuple(chem_comp.is_leaving_atom),
            tuple(map(tuple, chem_comp.bonds.as_array()[:, :2])),
        )
    )
