import torch
import numpy

from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import KinematicModuleData
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.scan_ordering import (
    construct_kin_module_data_for_pose,
)


def test_movemap_construction_smoke(stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    ff_edges_cpu = ff_2ubq_6res_H

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap(
        pose_stack.n_poses,
        pose_stack.max_n_blocks,
        pbt.max_n_torsions,
        pose_stack.max_n_block_atoms,
        pose_stack.device,
    )
