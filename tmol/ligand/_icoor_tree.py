"""Atom trees and Rosetta-convention internal coordinates for building residue types."""

import math
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np


def vertex_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """The angle at vertex ``b``, in radians."""
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def signed_dihedral_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """The dihedral ``a-b-c-d`` in radians, signed opposite to IUPAC's."""
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    return float(np.arctan2(float(np.dot(m1, n2)), float(np.dot(n1, n2))))


def icoor_geometry_from_coords(
    coords: np.ndarray,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
) -> np.ndarray:
    """``(d, theta, phi)`` per atom of ``order``, measured from ``coords``; the root is
    zeros and the next two atoms take ``theta = pi``, ``phi = 0``."""
    geometry = np.zeros((len(order), 3))
    for position, index in enumerate(order):
        atom, par = coords[index], coords[parent[index]]
        grandparent, great_grandparent = (coords[i] for i in grandparents[index])
        d = float(np.linalg.norm(atom - par)) if position >= 1 else 0.0
        theta = (
            math.pi - vertex_angle(atom, par, grandparent)
            if position >= 2
            else (math.pi if position else 0.0)
        )
        phi = (
            -signed_dihedral_angle(atom, par, grandparent, great_grandparent)
            if position >= 3
            else 0.0
        )
        geometry[position] = d, theta, phi
    return geometry


def build_atom_tree(
    n_atoms: int,
    bonds: Iterable[tuple[int, int]],
    is_heavy: Sequence[bool],
    root: int,
    frame_excluded_indices: Iterable[int] = (),
) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    """``(order, parent, grandparents)``: heavy atoms breadth-first from ``root``, then
    hydrogens; siblings stand in for missing ancestors, frame-excluded atoms go last.
    """
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    visited = [False] * n_atoms
    parent: dict[int, int] = {root: root}
    order: list[int] = []
    queue: deque[int] = deque([root])
    visited[root] = True
    priority = (
        (lambda i: (i in frame_excluded_indices, i)) if frame_excluded_indices else None
    )
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbour in sorted(adjacency[current], key=priority):
            if visited[neighbour] or not is_heavy[neighbour]:
                continue
            visited[neighbour] = True
            parent[neighbour] = current
            queue.append(neighbour)
    for heavy_index in list(order):
        for neighbour in sorted(adjacency[heavy_index]):
            if not visited[neighbour]:
                visited[neighbour] = True
                parent[neighbour] = heavy_index
                order.append(neighbour)

    position = {index: i for i, index in enumerate(order)}

    def pick(of: int, exclude: set[int], heavy_only: bool, before: int) -> int | None:
        """The earliest-placed neighbour of ``of`` before ``before``."""
        candidates = [
            n
            for n in adjacency[of]
            if n not in exclude and position.get(n, len(order)) < before
        ]
        if heavy_only and any(is_heavy[n] for n in candidates):
            candidates = [n for n in candidates if is_heavy[n]]
        return min(candidates, key=lambda n: position.get(n, len(order)), default=None)

    grandparents: dict[int, tuple[int, int]] = {}
    for index in order:
        index_heavy = bool(is_heavy[index])
        par = parent[index]
        before = position[index]
        grandparent = parent.get(par, par)
        if grandparent == par and index != root:
            substitute = pick(par, {index, par}, index_heavy, before)
            if substitute is not None:
                grandparent = substitute
        great_grandparent = parent.get(grandparent, grandparent)
        if not index_heavy and index != root:
            sibling = pick(par, {index, grandparent}, True, before)
            if sibling is not None:
                great_grandparent = sibling
        if great_grandparent in (grandparent, par, index) and index != root:
            substitute = pick(par, {index, par, grandparent}, index_heavy, before)
            if substitute is None:
                substitute = pick(
                    grandparent, {par, grandparent, index}, index_heavy, before
                )
            if substitute is not None:
                great_grandparent = substitute
        grandparents[index] = (grandparent, great_grandparent)
    return order, parent, grandparents


def find_root_atom(
    coords: np.ndarray,
    bonds: Iterable[tuple[int, int]],
    is_heavy: Sequence[bool],
    skip_indices: set[int] | None = None,
) -> int:
    """The heavy atom outside ``skip_indices`` nearest the centre with two heavy
    neighbours, else the first such heavy atom."""
    skip_indices = skip_indices or set()
    n_atoms = len(is_heavy)
    heavy_degree = [0] * n_atoms
    for i, j in bonds:
        i, j = int(i), int(j)
        heavy_degree[i] += bool(is_heavy[j])
        heavy_degree[j] += bool(is_heavy[i])
    distances_sq = np.sum((coords - coords.mean(axis=0)) ** 2, axis=1)
    candidates = [
        i
        for i in range(n_atoms)
        if i not in skip_indices and is_heavy[i] and heavy_degree[i] >= 2
    ]
    if candidates:
        return min(candidates, key=lambda i: distances_sq[i])
    for i in range(n_atoms):
        if i not in skip_indices and is_heavy[i]:
            return i
    raise ValueError("No valid root atom found (no heavy atoms)")
