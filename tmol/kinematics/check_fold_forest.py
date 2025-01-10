import numpy
import numba

from tmol.types.array import NDArray
from tmol.kinematics.fold_forest import EdgeType


@numba.jit(nopython=True)
def mark_polymeric_bonds_in_foldforest_edges(
    n_poses: int,
    max_n_blocks: int,
    n_blocks: NDArray[int][:],
    edges: NDArray[int][:, :, 4],
):
    """Make each implicit i-to-i+1 or i-to-(i-1) polymer bond explicit

    Notes
    -----
    This code does not ensure that the polymeric bonds between
    these two residues are present in the PoseStack; this means
    that if there are missing loops, e.g., that we can still
    "fold through" them.
    """
    polymeric_connection_in_edge = numpy.zeros(
        (n_poses, max_n_blocks, max_n_blocks), dtype=numpy.int64
    )
    max_n_edges = edges.shape[1]
    bad_edges = numpy.full((n_poses, max_n_edges), -1, dtype=numpy.int64)
    count_bad_for_pose = numpy.full((n_poses,), 0, dtype=numpy.int64)
    for i in range(n_poses):
        count_bad = 0
        for j in range(edges.shape[1]):
            if edges[i, j, 1] >= n_blocks[i] or edges[i, j, 2] >= n_blocks[i]:
                bad_edges[i, count_bad] = j
                count_bad += 1
                continue
            if edges[i, j, 0] == EdgeType.polymer:
                increment = 1 if edges[i, j, 1] < edges[i, j, 2] else -1

                for k in range(edges[i, j, 1], edges[i, j, 2], increment):
                    polymeric_connection_in_edge[i, k, k + increment] += 1
        count_bad_for_pose[i] = count_bad

    return (polymeric_connection_in_edge, count_bad_for_pose, bad_edges)


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
def ensure_jumps_numbered_and_distinct(
    edges: NDArray[numpy.int64][:, :, 4],
):
    n_poses = edges.shape[0]
    max_n_edges = edges.shape[1]
    jump_numbers = numpy.full((n_poses, max_n_edges), -1, dtype=numpy.int64)
    count_n_jumps = numpy.zeros((n_poses,), dtype=numpy.int64)
    found_bad_jump = False
    bad_jump_numbers = numpy.full((n_poses, max_n_edges), -1, dtype=numpy.int64)
    count_n_bad_jumps = numpy.zeros((n_poses,), dtype=numpy.int64)
    for i in range(n_poses):
        for j in range(max_n_edges):
            if edges[i, j, 0] == EdgeType.jump:
                count_n_jumps[i] += 1
                if edges[i, j, 3] < 0 or edges[i, j, 3] >= max_n_edges:
                    found_bad_jump = True
                    bad_jump_ind = count_n_bad_jumps[i]
                    bad_jump_numbers[i, bad_jump_ind] = j
                    count_n_bad_jumps[i] += 1
                    continue
                if jump_numbers[i, edges[i, j, 3]] != -1:
                    # this jump number has already been seen
                    found_bad_jump = True
                    bad_jump_ind = count_n_bad_jumps[i]
                    bad_jump_numbers[i, bad_jump_ind] = j
                    count_n_bad_jumps[i] += 1
                    continue
                jump_numbers[i, edges[i, j, 3]] = j
        # now, we look for jumps with indices >= the number of jumps
        # that we actually counted; a fold tree with such a jump must
        # have non-contiguous indices starting from 0.
        for j in range(count_n_jumps[i], max_n_edges):
            if jump_numbers[i, j] != -1:
                found_bad_jump = True
                bad_jump_ind = count_n_bad_jumps[i]
                bad_jump_numbers[i, bad_jump_ind] = jump_numbers[i, j]
                count_n_bad_jumps[i] += 1
    return found_bad_jump, count_n_bad_jumps, bad_jump_numbers, count_n_jumps


@numba.jit(nopython=True)
def validate_fold_forest_jit(
    roots: NDArray[numpy.int64][:],
    n_blocks: NDArray[numpy.int64][:],
    edges: NDArray[numpy.int64][:, :, 4],
):
    n_poses = n_blocks.shape[0]
    max_n_blocks = n_blocks.max()
    max_n_edges = edges.shape[1]
    connections, count_bad, bad_edges = mark_polymeric_bonds_in_foldforest_edges(
        n_poses, max_n_blocks, n_blocks, edges
    )
    error = False
    for i in range(n_poses):
        if count_bad[i] > 0:
            error = True
    if error:
        return False, bad_edges, None, None, None, None, None

    # ok, let's get the other edges incorporated
    for i in range(n_poses):
        for j in range(max_n_edges):
            if edges[i, j, 0] == EdgeType.jump or edges[i, j, 0] == EdgeType.chemical:
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

    found_bad_jump, count_n_bad_jumps, bad_jump_numbers, count_n_jumps = (
        ensure_jumps_numbered_and_distinct(edges)
    )
    good = good and not found_bad_jump

    return (
        good,
        bad_edges,
        cycles_detected,
        missing,
        count_n_bad_jumps,
        bad_jump_numbers,
        count_n_jumps,
    )


def validate_fold_forest(
    roots: NDArray[numpy.int64][:],
    n_blocks: NDArray[numpy.int64][:],
    edges: NDArray[numpy.int64][:, :, 4],
):
    (
        good,
        bad_edges,
        cycles_detected,
        missing,
        count_n_bad_jumps,
        bad_jump_numbers,
        count_n_jumps,
    ) = validate_fold_forest_jit(roots, n_blocks, edges)

    if not good:
        n_poses = n_blocks.shape[0]
        max_n_edges = edges.shape[1]
        errors = []
        for i in range(n_poses):
            for j in range(max_n_edges):
                if bad_edges[i, j] == -1:
                    # bad edges are listed first, so
                    # if we hit "-1", there are none remaining
                    break
                edge_index = bad_edges[i, j]
                edge_start = edges[i, edge_index, 1]
                edge_end = edges[i, edge_index, 2]
                if edge_start >= n_blocks[i]:
                    errors.append(
                        " ".join(
                            [
                                f"FOLD FOREST ERROR: Bad edge {edge_index} in pose {i}",
                                f"gives start index {edge_start} out of range; (n_blocks[{i}] = {n_blocks[i]})",
                            ]
                        )
                    )
                if edge_end >= n_blocks[i]:
                    errors.append(
                        " ".join(
                            [
                                f"FOLD FOREST ERROR: Bad edge {edge_index} in pose {i}",
                                f"gives end index {edge_end} out of range; (n_blocks[{i}] = {n_blocks[i]})",
                            ]
                        )
                    )

        for i in range(n_poses):
            if cycles_detected is None:
                break
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
        for i in range(n_poses):
            if count_n_bad_jumps is None:
                break
            if count_n_bad_jumps[i] > 0:
                for j in range(count_n_bad_jumps[i]):
                    e = edges[i, bad_jump_numbers[i, j], :]
                    is_repeat_index = False
                    first_edge_w_index = -1
                    for k in range(bad_jump_numbers[i, j]):
                        # print(f"e: {e[0]}, {e[1]}, {e[2]}, {e[3]} and k: {k} edge {edges[i, k, 0]}, {edges[i, k, 3]}")
                        if edges[i, k, 0] == EdgeType.jump and edges[i, k, 3] == e[3]:
                            is_repeat_index = True
                            first_edge_w_index = k
                            break
                    if is_repeat_index:
                        ek = edges[i, first_edge_w_index, :]
                        errors.append(
                            " ".join(
                                [
                                    "FOLD FOREST ERROR: Jump",
                                    f"[p={e[0]}, s={e[1]}, e={e[2]}, ind={e[3]}]",
                                    "in pose",
                                    str(i),
                                    "has repeated jump index with edge",
                                    str(first_edge_w_index),
                                    f"[p={ek[0]}, s={ek[1]}, e={ek[2]}, ind={ek[3]}]",
                                ]
                            )
                        )
                    else:
                        if e[3] < 0:
                            errors.append(
                                " ".join(
                                    [
                                        "FOLD FOREST ERROR: Jump",
                                        f"[p={e[0]}, s={e[1]}, e={e[2]}, ind={e[3]}]",
                                        "in pose",
                                        str(i),
                                        "has negative jump index",
                                    ]
                                )
                            )
                        else:
                            errors.append(
                                " ".join(
                                    [
                                        "FOLD FOREST ERROR: Jump",
                                        f"[p={e[0]}, s={e[1]}, e={e[2]}, ind={e[3]}]",
                                        "in pose",
                                        str(i),
                                        "has a non-contiguous-starting-at-0 jump index",
                                        f"(n jumps total: {count_n_jumps[i]})",
                                    ]
                                )
                            )
        raise ValueError("\n".join(errors))
