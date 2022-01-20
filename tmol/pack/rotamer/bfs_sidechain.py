import numpy
import numba

from typing import List

from tmol.types.functional import validate_args
from tmol.chemical.restypes import RefinedResidueType


@numba.jit(nopython=True)
def bfs_sidechain_atoms_jit(parents, sc_roots):
    n_atoms = parents.shape[0]
    n_children = numpy.zeros(n_atoms, dtype=numpy.int32)
    for i in range(n_atoms):
        n_children[parents[i]] += 1
    children_start = numpy.concatenate(
        (numpy.zeros(1, dtype=numpy.int32), numpy.cumsum(n_children)[:-1])
    )

    child_count = numpy.zeros(n_atoms, dtype=numpy.int32)
    children = numpy.full(n_atoms, -1, dtype=numpy.int32)
    for i in range(n_atoms):
        parent = parents[i]
        children[children_start[parent] + child_count[parent]] = i
        child_count[parent] += 1

    bfs_count_end = 0
    bfs_curr = 0
    bfs_list = numpy.full(n_atoms, -1, dtype=numpy.int32)
    visited = numpy.zeros(n_atoms, dtype=numpy.int32)
    for root in sc_roots:
        # put the root children in the bfs list
        visited[root] = 1
        for child_ind in range(
            children_start[root], children_start[root] + child_count[root]
        ):
            bfs_list[bfs_count_end] = children[child_ind]
            bfs_count_end += 1
        while bfs_curr != bfs_count_end:
            node = bfs_list[bfs_curr]
            bfs_curr += 1
            if visited[node]:
                # can happen when the root of the kintree is given
                # as a sidechain root
                continue
            # add node's children to the bfs_list
            for child_ind in range(
                children_start[node], children_start[node] + child_count[node]
            ):
                bfs_list[bfs_count_end] = children[child_ind]
                bfs_count_end += 1
            visited[node] = 1
    return visited


@validate_args
def bfs_sidechain_atoms(restype: RefinedResidueType, sc_roots: List):
    # first descend through the sidechain
    id = restype.rotamer_kintree.id
    parents = restype.rotamer_kintree.parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]
    return bfs_sidechain_atoms_jit(parents, numpy.array(sc_roots, dtype=numpy.int32))
