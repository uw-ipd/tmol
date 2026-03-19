import torch

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.kinematics.fold_forest import FoldForest, EdgeType


def _real_edges(fold_forest, pose_idx):
    """Return the set of (type, start, end) tuples for all real edges in a pose."""
    return {
        (EdgeType(int(e[0])), int(e[1]), int(e[2]))
        for e in fold_forest.edges[pose_idx]
        if e[0] != -1
    }


def _check_jump_indices(fold_forest, pose_idx):
    """Assert that jump indices form a valid 0..n_jumps-1 assignment."""
    n_e = fold_forest.n_edges[pose_idx]
    jump_indices = [
        int(fold_forest.edges[pose_idx, j, 3])
        for j in range(n_e)
        if fold_forest.edges[pose_idx, j, 0] in (EdgeType.root_jump, EdgeType.jump)
    ]
    assert sorted(jump_indices) == list(range(len(jump_indices)))


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

    print(_real_edges(fold_forest, 1))

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
