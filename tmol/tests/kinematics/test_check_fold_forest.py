import numpy

from tmol.kinematics.check_fold_forest import (
    mark_polymeric_bonds_in_foldforest_edges,
    bfs_proper_forest,
    validate_fold_forest,
)

from tmol.kinematics.fold_forest import EdgeType


def test_mark_polymeric_bonds_in_foldforest_edges_1():
    n_res_per_tree = numpy.array([8, 11, 5], dtype=int)

    max_n_edges = 2
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.polymer
    edges[:, 0, 1] = 0
    edges[:, 0, 2] = n_res_per_tree - 1
    edges[:, 1, 0] = EdgeType.root_jump
    edges[:, 1, 1] = -1
    edges[:, 1, 2] = 0

    polymeric_connection_in_edge, count_bad_edges, bad_edges = (
        mark_polymeric_bonds_in_foldforest_edges(3, 11, n_res_per_tree, edges)
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
    for pid, r1, r2 in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1
    count_bad_edges_gold = numpy.zeros((3,), dtype=numpy.int64)
    bad_edges_gold = numpy.full((3, max_n_edges), -1, dtype=numpy.int64)

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)
    numpy.testing.assert_equal(count_bad_edges_gold, count_bad_edges)
    numpy.testing.assert_equal(bad_edges_gold, bad_edges)


def test_mark_polymeric_bonds_in_foldforest_edges_2():
    # ok, pose 1 will be rooted at residue 5 and have two polymer edges
    # from the root to its termini
    max_n_edges = 3
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.root_jump
    edges[:, 0, 1] = -1
    edges[:, 0, 2] = 0
    edges[1, 0, 2] = 5

    edges[:, 1, 0] = EdgeType.polymer
    edges[:, 1, 1] = 0
    edges[0, 1, 2] = 7
    edges[1, 1, 1] = 5
    edges[1, 1, 2] = 0
    edges[1, 2, 0] = EdgeType.polymer
    edges[1, 2, 1] = 5
    edges[1, 2, 2] = 10
    edges[2, 1, 2] = 4
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    polymeric_connection_in_edge, _1, _2 = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, n_res_per_tree, edges
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
    for pid, r1, r2 in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)


def test_mark_polymeric_bonds_in_foldforest_edges_3():
    # n_res_per_tree = [8, 11, 5]

    # ok, this time pose 2 will have a cutpoint at residue 5
    max_n_edges = 4
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.root_jump
    edges[:, 0, 1] = -1
    edges[:, 0, 2] = 0

    edges[:, 1, 0] = EdgeType.polymer
    edges[:, 1, 1] = 0
    edges[0, 1, 2] = 7
    edges[1, 1, 1] = 0
    edges[1, 1, 2] = 5
    edges[1, 2, 0] = EdgeType.polymer
    edges[1, 2, 1] = 8
    edges[1, 2, 2] = 6
    edges[1, 3, 0] = EdgeType.polymer
    edges[1, 3, 1] = 8
    edges[1, 3, 2] = 10
    edges[2, 1, 2] = 4

    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    polymeric_connection_in_edge, _1, _2 = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, n_res_per_tree, edges
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
    for pid, r1, r2 in polymeric_edges:
        polymeric_connections_gold[pid, r1, r2] = 1

    numpy.testing.assert_equal(polymeric_connections_gold, polymeric_connection_in_edge)


def test_bfs_proper_forest_1():
    roots = numpy.array([[0], [0], [0]], dtype=numpy.int64)
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
    for pid, r1, r2 in edges:
        connections[pid, r1, r2] = 1

    cycles_detected, missing = bfs_proper_forest(roots, n_res_per_tree, connections)

    numpy.testing.assert_equal(numpy.zeros((3, 2), dtype=numpy.int64), cycles_detected)
    numpy.testing.assert_equal(numpy.zeros((3, 11), dtype=numpy.int64), missing)


def test_bfs_proper_forest_2():
    roots = numpy.array([[0], [0], [0]], dtype=numpy.int64)
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
    for pid, r1, r2 in edges:
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


def count_max_n_edges(edges_compact):
    max_n_poses = edges_compact[-1][0] + 1
    n_edges = numpy.zeros((max_n_poses,), dtype=numpy.int64)
    for pid, _, _, _, _ in edges_compact:
        n_edges[pid] += 1
    return numpy.max(n_edges)


def test_validate_fold_forest_1():
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 7, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.polymer, 8, 6, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        print(verr)
        assert verr is None


def test_validate_fold_forest_2():
    """Make sure that if a node is unreachable, in this case node 6 in tree 1,
    that the validate_fold_tree function throws an exception
    """
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 7, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        assert verr.args[0] == "FOLD FOREST ERROR: Block 6 unreachable in pose 1"
        threw = True
    assert threw


def test_validate_fold_forest_2b():
    """Make sure that if a node is unreachable, in this case node 4 in tree 1,
    that the validate_fold_tree function throws an exception.
    The oopsie: the user "meant" to make the 5->3 edge a peptide edge
    but labled it a jump and has now skipped block 4.
    """
    n_res_per_tree = numpy.array([6, 6], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 2, -1),
        (0, EdgeType.polymer, 2, 0, -1),
        (0, EdgeType.jump, 2, 5, 0),
        (0, EdgeType.polymer, 5, 3, -1),
        (1, EdgeType.root_jump, -1, 5, -1),
        (1, EdgeType.polymer, 2, 0, -1),
        (1, EdgeType.jump, 5, 2, 0),
        (1, EdgeType.jump, 5, 3, 1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((2, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        assert verr.args[0] == "FOLD FOREST ERROR: Block 4 unreachable in pose 1"
        threw = True
    assert threw


def test_validate_fold_forest_2c():
    """Another version of testing if edges are listed in that are not part of the Pose"""
    # roots = numpy.array([2, 4, 4], dtype=numpy.int64)
    n_res_per_tree = numpy.array([4, 5, 6], dtype=numpy.int64)

    # in this case, we have too many residues for pose 1 and too few for pose 2
    edges_compact = [
        (0, EdgeType.root_jump, -1, 2, -1),
        (0, EdgeType.polymer, 1, 0, -1),
        (0, EdgeType.polymer, 1, 2, -1),
        (0, EdgeType.jump, 1, 3, 0),
        (1, EdgeType.root_jump, -1, 4, -1),
        (1, EdgeType.polymer, 1, 0, -1),
        (1, EdgeType.polymer, 1, 2, -1),
        (1, EdgeType.jump, 4, 1, 0),
        (1, EdgeType.polymer, 4, 3, -1),
        (1, EdgeType.polymer, 4, 5, -1),
        (2, EdgeType.root_jump, -1, 4, -1),
        (2, EdgeType.polymer, 1, 0, -1),
        (2, EdgeType.polymer, 1, 2, -1),
        (2, EdgeType.jump, 4, 1, 0),
        (2, EdgeType.polymer, 4, 3, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        # print(verr)
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Bad edge 5 in pose 1 gives end index 5 out of range; (n_blocks[1] = 5)"
        )
        threw = True
    assert threw


def test_validate_fold_forest_3():
    """Make sure that if two trees have errors, that both errors are reported"""
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([8, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 7, -1),
        (0, EdgeType.polymer, 6, 3, -1),  # extra edge
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.polymer, 8, 5, -1),  # edge goes too far to block 5
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        threw = True
        gold_error = (
            "FOLD FOREST ERROR: Cycle detected in pose 0 at block 3\n"
            "FOLD FOREST ERROR: Cycle detected in pose 1 at block 5"
        )
        assert verr.args[0] == gold_error
    assert threw


def test_validate_fold_forest_4():
    """Make sure that if there are more nodes than residues, in this case node 7 in tree 0
    that the validate_fold_tree function throws an exception
    """
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([6, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 7, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 6, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Bad edge 1 in pose 0 gives end index 7 out of range; (n_blocks[0] = 6)"
        )
        threw = True
    assert threw


def test_validate_fold_forest_5():
    """Make sure that jumps are given different ids."""
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([6, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 3, -1),
        (1, EdgeType.polymer, 6, 4, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.jump, 0, 6, 0),  # Error: duplicate jump id
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        # print(verr)
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Jump [p=1, s=0, e=6, ind=0] in pose 1 has repeated jump index with edge 3 [p=1, s=0, e=8, ind=0]"
        )
        threw = True
    assert threw


def test_validate_fold_forest_6():
    """Make sure that jumps indices are non-negative."""
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([6, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 3, -1),
        (1, EdgeType.polymer, 6, 4, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.jump, 0, 6, -1),  # Error: negative jump id
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        # print(verr)
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Jump [p=1, s=0, e=6, ind=-1] in pose 1 has negative jump index"
        )
        threw = True
    assert threw


def test_validate_fold_forest_7():
    """Make sure that jumps indices are contiguous starting at 0."""
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([6, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 3, -1),
        (1, EdgeType.polymer, 6, 4, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.jump, 0, 6, 2),  # Error: jump id == n-jumps
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        # print(verr)
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Jump [p=1, s=0, e=6, ind=2] in pose 1 has a non-contiguous-starting-at-0 jump index (n jumps total: 2)"
        )
        threw = True
    assert threw


def test_validate_fold_forest_7b():
    """Make sure that if jumps are given different ids."""
    # roots = numpy.array([0, 0, 0], dtype=numpy.int64)
    n_res_per_tree = numpy.array([6, 11, 5], dtype=numpy.int64)

    edges_compact = [
        (0, EdgeType.root_jump, -1, 0, -1),
        (0, EdgeType.polymer, 0, 5, -1),
        (1, EdgeType.root_jump, -1, 0, -1),
        (1, EdgeType.polymer, 0, 3, -1),
        (1, EdgeType.polymer, 6, 4, -1),
        (1, EdgeType.jump, 0, 8, 0),
        (1, EdgeType.jump, 0, 6, 6),  # Error: jump id >= n-edges
        (1, EdgeType.polymer, 8, 7, -1),
        (1, EdgeType.polymer, 8, 10, -1),
        (2, EdgeType.root_jump, -1, 0, -1),
        (2, EdgeType.polymer, 0, 4, -1),
    ]
    max_n_edges = count_max_n_edges(edges_compact)
    count_pose_edges = numpy.zeros((3,), dtype=numpy.int64)
    edges = numpy.full((3, max_n_edges, 4), -1, dtype=numpy.int64)
    for pid, edge_type, r1, r2, jid in edges_compact:
        edges[pid, count_pose_edges[pid], 0] = edge_type
        edges[pid, count_pose_edges[pid], 1] = r1
        edges[pid, count_pose_edges[pid], 2] = r2
        edges[pid, count_pose_edges[pid], 3] = jid
        count_pose_edges[pid] += 1

    threw = False
    try:
        validate_fold_forest(n_res_per_tree, edges)
    except ValueError as verr:
        # print(verr)
        assert (
            verr.args[0]
            == "FOLD FOREST ERROR: Jump [p=1, s=0, e=6, ind=6] in pose 1 has a non-contiguous-starting-at-0 jump index (n jumps total: 2)"
        )
        threw = True
    assert threw
