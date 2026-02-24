import time

import torch

from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MinimizerMap, MoveMap
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction


def build_kinforest_network(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
):
    from tmol.kinematics.script_modules import PoseStackKinematicsModule
    from tmol.optimization.sfxn_modules import KinForestSfxnNetwork

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    kin_module = PoseStackKinematicsModule(pose_stack, ff)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()

    minimizer_map = MinimizerMap(pose_stack, kin_module.kmd, mm)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    kf_network = KinForestSfxnNetwork(sfxn, pose_stack, kin_module, minimizer_map.dof_mask)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"build_kinforest_network {end_time3 - start_time: .2f}"
            + f" s1: {end_time1 - start_time: .2f} s2: {end_time2 - end_time1: .2f}"
            + f" s3: {end_time3 - end_time2: .2f}"
        )

    return kf_network


def run_kin_min(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
):
    """Run LBFGS minimization on a PoseStack in internal DOF space.

    Note that this function constructs a new kinematics module and discards it
    at the conclusion of minimization.
    """
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    kf_network = build_kinforest_network(pose_stack, sfxn, ff, mm, verbose)
    optimizer = LBFGS_Armijo(kf_network.parameters())

    def closure():
        optimizer.zero_grad()
        E = kf_network().sum()
        E.backward()
        return E

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()
    optimizer.step(closure)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    new_pose_stack = kf_network.pose_stack_from_dofs()
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"kin_min {end_time3 - start_time: .2f} setup: {end_time1 - start_time: .2f}"
            + f" opt {end_time2 - end_time1: .2f} stack-ctor: {end_time3 - end_time2: .2f}"
        )

    return new_pose_stack
