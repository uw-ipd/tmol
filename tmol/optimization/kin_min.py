# import torch
# import numpy

from tmol.pose.pose_stack import PoseStack

# from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.score.score_function import ScoreFunction

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo


def build_kinforest_network(
    pose_stack: PoseStack, sfxn: ScoreFunction, ff: FoldForest, mm: MoveMap
):
    from tmol.kinematics.script_modules import PoseStackKinematicsModule
    from tmol.optimization.sfxn_modules import KinForestSfxnNetwork

    kin_module = PoseStackKinematicsModule(pose_stack, ff)
    minimizer_map = MinimizerMap(pose_stack, kin_module.kmd, mm)
    kf_network = KinForestSfxnNetwork(
        sfxn, pose_stack, kin_module, minimizer_map.dof_mask
    )
    return kf_network


def run_kin_min(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
):
    """Run LBFGS minimization on a PoseStack in internal DOF space.

    Note that this function constructs a new kinematics module and discards it
    at the conclusion of minimization.
    """
    kf_network = build_kinforest_network(pose_stack, sfxn, ff, mm)
    optimizer = LBFGS_Armijo(kf_network.parameters())

    def closure():
        optimizer.zero_grad()
        E = kf_network().sum()
        E.backward()
        return E

    optimizer.step(closure)

    return kf_network.pose_stack_from_dofs()
