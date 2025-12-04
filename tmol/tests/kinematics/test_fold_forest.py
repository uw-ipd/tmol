import torch
import numpy

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.kinematics.fold_forest import FoldForest, EdgeType


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
    # lets pretend residues 100 and 101 are connected.
    p2 = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6
    assert fold_forest.n_edges[0] == 2
    assert fold_forest.n_edges[1] == 6

    ff_edges_gold = [
        [
            [EdgeType.polymer, 0, 75, -1],
            [EdgeType.root_jump, -1, 0, 0],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        [
            [EdgeType.polymer, 0, 554, -1],
            [EdgeType.root_jump, -1, 0, 0],
            [EdgeType.polymer, 555, 768, -1],
            [EdgeType.root_jump, -1, 555, 1],
            [EdgeType.polymer, 769, 990, -1],
            [EdgeType.root_jump, -1, 769, 2],
        ],
    ]
    ff_edges_gold = numpy.array(ff_edges_gold)
    numpy.testing.assert_equal(fold_forest.edges, ff_edges_gold)
