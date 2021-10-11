import numpy
import numba

from tmol.types.array import NDArray
from tmol.kinematics.fold_forest import EdgeType


@numba.jit(nopython=True)
def mark_polymeric_bonds_in_foldforest_edges(
    n_poses: int, max_n_blocks: int, edges: NDArray[int][:, :, 4]
):
    polymeric_connection_in_edge = numpy.zeros(
        (n_poses, max_n_blocks, max_n_blocks), dtype=numpy.int64
    )
    for i in range(n_poses):
        for j in range(edges.shape[1]):
            if edges[i, j, 0] == EdgeType.polymer:
                increment = 1 if edges[i, j, 1] < edges[i, j, 2] else -1

                for k in range(edges[i, j, 1], edges[i, j, 2], increment):
                    polymeric_connection_in_edge[i, k, k + increment] += 1

    return polymeric_connection_in_edge


@numba.jit(nopython=True)
def bfs_proper_forest(
    roots: NDArray[numpy.int64][:],
    n_blocks: NDArray[numpy.int64][:],
    connections: NDArray[numpy.int64][:, :, :],
):
    n_poses = connections.shape[0]
    max_n_blocks = connections.shape[1]
    assert n_blocks.shape[0] == n_poses
    assert roots.shape[0] == n_poses

    cycles_detected = numpy.zeros((n_poses, 2), dtype=numpy.int64)
    missing = numpy.zeros((n_poses, max_n_blocks), dtype=numpy.int64)

    for i in range(n_poses):
        found_cycle = False
        node_colors = numpy.zeros((max_n_blocks,), dtype=numpy.int64)
        visit_queue = numpy.full((max_n_blocks,), -1, dtype=numpy.int64)
        visit_front = 0
        visit_back = 0
        visit_queue[visit_front] = roots[i]
        node_colors[roots[i]] = 1
        visit_back += 1
        while visit_front < visit_back and not found_cycle:
            focused = visit_queue[visit_front]
            for j in range(max_n_blocks):
                if connections[i, focused, j] == 0:
                    continue
                if connections[i, focused, j] > 1:
                    found_cycle = True
                    cycles_detected[i, 0] = 1
                    cycles_detected[i, 1] = j
                    break
                if node_colors[j] == 1:
                    found_cycle = True
                    cycles_detected[i, 0] = 1
                    cycles_detected[i, 1] = j
                    break
                node_colors[j] = 1
                visit_queue[visit_back] = j
                visit_back += 1
            visit_front += 1
        for j in range(n_blocks[i]):
            if node_colors[j] != 1:
                missing[i, j] = 1
    return cycles_detected, missing


@numba.jit(nopython=True)
def validate_fold_forest_jit(
    roots: NDArray[numpy.int64][:],
    n_blocks: NDArray[numpy.int64][:],
    edges: NDArray[numpy.int64][:, :, 4],
):
    n_poses = n_blocks.shape[0]
    max_n_blocks = n_blocks.max()
    max_n_edges = edges.shape[2]
    connections = mark_polymeric_bonds_in_foldforest_edges(n_poses, max_n_blocks, edges)

    # ok, let's get the other edges incorporated
    for i in range(n_poses):
        for j in range(max_n_edges):
            if edges[i, j, 0] == EdgeType.jump:
                r1 = edges[i, j, 1]
                r2 = edges[i, j, 2]
                connections[i, r1, r2] += 1
            if edges[i, j, 0] == EdgeType.chemical:
                r1 = edges[i, j, 1]
                r2 = edges[i, j, 2]
                connections[i, r1, r2] += 1

    # and now a BFS
    cycles_detected, missing = bfs_proper_forest(roots, n_blocks, connections)

    good = True
    for i in range(n_poses):
        if cycles_detected[i, 0] != 0:
            good = False
            break
        for j in range(n_blocks[i]):
            if missing[i, j] == 1:
                good = False
                break
        if not good:
            break

    return good, cycles_detected, missing


def validate_fold_forest(
    roots: NDArray[numpy.int64][:],
    n_blocks: NDArray[numpy.int64][:],
    edges: NDArray[numpy.int64][:, :, 4],
):
    good, cycles_detected, missing = validate_fold_forest_jit(roots, n_blocks, edges)

    if not good:
        n_poses = n_blocks.shape[0]
        errors = []
        for i in range(n_poses):
            if cycles_detected[i, 0] != 0:
                good = False
                errors.append(
                    " ".join(
                        [
                            "FOLD FOREST ERROR: Cycle detected in pose",
                            str(i),
                            "at block",
                            str(cycles_detected[i, 1]),
                        ]
                    )
                )
                continue
            for j in range(n_blocks[i]):
                if missing[i, j] == 1:
                    good = False
                    errors.append(
                        " ".join(
                            [
                                "FOLD FOREST ERROR: Block",
                                str(j),
                                "unreachable in pose",
                                str(i),
                            ]
                        )
                    )
        raise ValueError("\n".join(errors))
