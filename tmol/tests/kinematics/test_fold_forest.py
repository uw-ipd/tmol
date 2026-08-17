import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.kinematics import validate_fold_forest
from tmol.kinematics import FoldForest, EdgeType, _build_pose_fold_forest


def _real_edges(fold_forest, pose_idx):
    """Return the set of (type, start, end) tuples for all real edges in a pose."""
    return {
        (EdgeType(int(e[0])), int(e[1]), int(e[2]))
        for e in fold_forest.edges[pose_idx]
        if e[0] != -1
    }


def _check_jump_indices(fold_forest, pose_idx):
    """Assert that jump indices form a valid 0..n_jumps-1 assignment.

    Only true jumps are numbered; a root jump is identified by its downstream
    block and carries -1, so numbering one would leave a gap in the jump
    indices that validate_fold_forest rejects.
    """
    n_e = fold_forest.n_edges[pose_idx]

    def indices_of(edge_type):
        return [
            int(fold_forest.edges[pose_idx, j, 3])
            for j in range(n_e)
            if fold_forest.edges[pose_idx, j, 0] == edge_type
        ]

    assert sorted(indices_of(EdgeType.jump)) == list(
        range(len(indices_of(EdgeType.jump)))
    )
    assert all(i == -1 for i in indices_of(EdgeType.root_jump))


def test_reasonable_fold_forest_smoke(default_database, erbb2_and_pertuzumab_pdb):
    torch_device = torch.device("cpu")
    p = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p], torch_device)

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6


def test_jagged_reasonable_fold_forest(
    ubq_pdb, erbb2_and_pertuzumab_pdb, default_database, dun_sampler, torch_device
):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    p2 = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6
    assert fold_forest.n_edges[0] == 2
    assert fold_forest.n_edges[1] == 6

    # Pose 0: ubiquitin — one polymer chain 0..75, one root-jump
    assert _real_edges(fold_forest, 0) == {
        (EdgeType.polymer, 0, 75),
        (EdgeType.root_jump, -1, 0),
    }
    _check_jump_indices(fold_forest, 0)

    # Pose 1: erbb2 + pertuzumab — three disconnected polymer chains
    assert _real_edges(fold_forest, 1) == {
        (EdgeType.polymer, 0, 554),
        (EdgeType.root_jump, -1, 0),
        (EdgeType.polymer, 555, 768),
        (EdgeType.root_jump, -1, 555),
        (EdgeType.polymer, 769, 990),
        (EdgeType.root_jump, -1, 769),
    }
    _check_jump_indices(fold_forest, 1)


def _linear_polymer_pose(segments, chain_ids):
    """Connectivity for one pose built from disjoint polymer segments.

    segments is a list of (first, last) inclusive residue ranges bonded
    up-to-down along the backbone; chain_ids gives each residue's biological
    chain. Block type 0 carries its down connection in slot 0 and its up
    connection in slot 1.
    """
    n_res = max(last for _, last in segments) + 1
    bti = numpy.zeros(n_res, dtype=numpy.int64)
    irc = numpy.full((n_res, 2, 2), -1, dtype=numpy.int64)
    for first, last in segments:
        for r in range(first, last):
            irc[r, 1] = (r + 1, 0)
            irc[r + 1, 0] = (r, 1)
    up_c = numpy.array([1], dtype=numpy.int64)
    down_c = numpy.array([0], dtype=numpy.int64)
    return bti, irc, up_c, down_c, numpy.array(chain_ids, dtype=numpy.int64)


def test_fold_forest_numbers_only_true_jumps():
    """A chain-internal break alongside separate chains must number contiguously.

    Residues 0-5 are one biological chain broken between 2 and 3, so the second
    segment is reached by a true jump; residues 6-8 are a second chain and are
    root-jumped. The true jump is emitted after a root jump, which is what used
    to push its index past the number of jumps.
    """
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose([(0, 2), (3, 5), (6, 8)], [0] * 6 + [1] * 3)
    )
    by_type = {}
    for edge_type, start, end, jump_ind in edges:
        by_type.setdefault(EdgeType(edge_type), []).append((start, end, jump_ind))

    assert by_type[EdgeType.root_jump] == [(-1, 0, -1), (-1, 6, -1)]
    assert by_type[EdgeType.jump] == [(2, 3, 0)]
    assert by_type[EdgeType.polymer] == [(0, 2, -1), (3, 5, -1), (6, 8, -1)]

    validate_fold_forest(
        numpy.array([9], dtype=numpy.int64),
        numpy.array([edges], dtype=numpy.int64),
    )
