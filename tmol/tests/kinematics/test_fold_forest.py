import torch
import numpy

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.kinematics.fold_forest import FoldForest


def test_reasonable_fold_forest_smoke(default_database, erbb2_and_pertuzumab_pdb):
    torch_device = torch.device("cpu")
    p = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p], torch_device)

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6
