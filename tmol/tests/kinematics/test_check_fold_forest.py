import numpy

from tmol.kinematics.check_fold_forest import (
    mark_polymeric_bonds_in_foldforest_edges,
    bfs_proper_forest,
    validate_fold_forest,
)

from tmol.kinematics.fold_forest import EdgeType


def test_mark_polymeric_bonds_in_foldforest_edges_1():
    n_res_per_tree = [8, 11, 5]

    edges = numpy.full((3, 1, 4), -1, dtype=int)
    edges[:, :, 0] = EdgeType.polymer
    edges[:, :, 1] = 0
    edges[:, 0, 2] = numpy.array(n_res_per_tree, dtype=int) - 1

    polymeric_connection_in_edge = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, edges
    )

    polymeric_connections_gold = numpy.zeros((3, 11, 11), dtype=numpy.int64)
    polymeric_edges = [
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
        (1, 4, 5),
        (1, 5, 6),
        (1, 6, 7),
        (1, 7, 8),
        (1, 8, 9),
        (1, 9, 10),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 4),
    ]
    for (pid, r1, r2) in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)


def test_mark_polymeric_bonds_in_foldforest_edges_2():
    n_res_per_tree = [8, 11, 5]

    # ok, pose 1 will be rooted at residue 5 and have two polymer edges
    # from the root to its termini
    edges = numpy.full((3, 2, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.polymer
    edges[:, 0, 1] = 0
    edges[0, 0, 2] = 7
    edges[1, 0, 1] = 5
    edges[1, 0, 2] = 0
    edges[1, 1, 0] = EdgeType.polymer
    edges[1, 1, 1] = 5
    edges[1, 1, 2] = 10
    edges[2, 0, 2] = 4

    polymeric_connection_in_edge = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, edges
    )

    polymeric_connections_gold = numpy.zeros((3, 11, 11), dtype=numpy.int64)
    polymeric_edges = [
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 2),
        (1, 4, 3),
        (1, 5, 4),
        (1, 5, 6),
        (1, 6, 7),
        (1, 7, 8),
        (1, 8, 9),
        (1, 9, 10),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 4),
    ]
    for (pid, r1, r2) in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)


def test_mark_polymeric_bonds_in_foldforest_edges_3():
    n_res_per_tree = [8, 11, 5]

    # ok, this time pose 2 will have a cutpoint at residue 5
    edges = numpy.full((3, 3, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.polymer
    edges[:, 0, 1] = 0
    edges[0, 0, 2] = 7
    edges[1, 0, 1] = 0
    edges[1, 0, 2] = 5
    edges[1, 1, 0] = EdgeType.polymer
    edges[1, 1, 1] = 8
    edges[1, 1, 2] = 6
    edges[1, 2, 0] = EdgeType.polymer
    edges[1, 2, 1] = 8
    edges[1, 2, 2] = 10
    edges[2, 0, 2] = 4

    polymeric_connection_in_edge = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, edges
    )

    polymeric_connections_gold = numpy.zeros((3, 11, 11), dtype=numpy.int64)
    polymeric_edges = [
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
        (1, 4, 5),
        # cutpoint! (1, 5, 6),
        (1, 7, 6),
        (1, 8, 7),
        (1, 8, 9),
        (1, 9, 10),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 4),
    ]
    for (pid, r1, r2) in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)


def test_bfs_proper_forest_1():
    roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    connections = numpy.zeros((3, 11, 11), dtype=numpy.int64)
    edges = [
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
        (1, 4, 5),
        (1, 0, 8),  # jump!
        (1, 7, 6),
        (1, 8, 7),
        (1, 8, 9),
        (1, 9, 10),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 4),
    ]
    for (pid, r1, r2) in edges:
        connections[pid, r1, r2] = 1

    cycles_detected, missing = bfs_proper_forest(roots, n_res_per_tree, connections)

    numpy.testing.assert_equal(numpy.zeros((3, 2), dtype=numpy.int64), cycles_detected)
    numpy.testing.assert_equal(numpy.zeros((3, 11), dtype=numpy.int64), missing)


def test_bfs_proper_forest_2():
    roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    connections = numpy.zeros((3, 11, 11), dtype=numpy.int64)
    edges = [
        (0, 0, 1),
        (0, 1, 3),  # detect cycle at residue 3
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
        (1, 0, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 3, 4),
        (1, 4, 5),
        (1, 0, 8),  # jump!
        # (1, 7, 6), # detect that residue 6 is unreached
        (1, 8, 7),
        (1, 8, 9),
        (1, 9, 10),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 4),
    ]
    for (pid, r1, r2) in edges:
        connections[pid, r1, r2] = 1

    cycles_detected, missing = bfs_proper_forest(roots, n_res_per_tree, connections)

    cycles_detected_gold = numpy.zeros((3, 2), dtype=numpy.int64)
    cycles_detected_gold[0, 0] = 1
    cycles_detected_gold[0, 1] = 3

    missing_gold = numpy.zeros((3, 11), dtype=numpy.int64)
    missing_gold[0, 4:8] = 1
    missing_gold[1, 6] = 1

    numpy.testing.assert_equal(cycles_detected_gold, cycles_detected)
    numpy.testing.assert_equal(missing_gold, missing)


def test_validate_fold_forest_1():
    roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.polymer, 0, 7),
        (1, EdgeType.polymer, 0, 5),
        (1, EdgeType.jump, 0, 8),
        (1, EdgeType.polymer, 8, 6),
        (1, EdgeType.polymer, 8, 10),
        (2, EdgeType.polymer, 0, 4),
    ]
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, 4, 4), -1, dtype=numpy.int64)
    for (pid, edge_type, r1, r2) in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        count_pose_edges[pid] += 1

    try:
        validate_fold_forest(roots, n_res_per_tree, edges)
        # we should reach here
    except ValueError as verr:
        print(verr)
        assert False


def test_validate_fold_forest_2():
    roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.polymer, 0, 7),
        (1, EdgeType.polymer, 0, 5),
        (1, EdgeType.jump, 0, 8),
        (1, EdgeType.polymer, 8, 7),
        (1, EdgeType.polymer, 8, 10),
        (2, EdgeType.polymer, 0, 4),
    ]
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, 4, 4), -1, dtype=numpy.int64)
    for (pid, edge_type, r1, r2) in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        count_pose_edges[pid] += 1

    try:
        validate_fold_forest(roots, n_res_per_tree, edges)
        # we should reach here
        assert False
    except ValueError as verr:
        assert verr.args[0] == "FOLD FOREST ERROR: Block 6 unreachable in pose 1"
