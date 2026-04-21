import time
import torch
from tmol.pose.pose_stack import PoseStack

from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.score.score_function import ScoreFunction

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo


def build_kinforest_network(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
    kin_dtype=torch.float32,
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

    kf_network = KinForestSfxnNetwork(
        sfxn, pose_stack, kin_module, minimizer_map.dof_mask, kin_dtype=kin_dtype
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"build_kinforest_network {end_time3 - start_time: .2f}"
            + f" s1: {end_time1-start_time: .2f} s2: {end_time2 - end_time1: .2f}"
            + f" s3: {end_time3-end_time2: .2f}"
        )

    return kf_network


def run_min(
    sfxn_module,
    optimizer_cls=LBFGS_Armijo,
    optimizer_kwargs=None,
    verbose=False,
):
    """Run minimization on any sfxn module (Cartesian or KinForest).

    The sfxn_module must be a torch.nn.Module whose forward() returns
    per-pose energies and which provides a pose_stack_from_dofs() method
    to extract the optimized PoseStack.

    Args:
        sfxn_module: A CartesianSfxnNetwork, KinForestSfxnNetwork, or
            any nn.Module with a compatible interface.
        optimizer_cls: A torch.optim.Optimizer class. Must support a
            closure-based step() call (e.g. LBFGS_Armijo, torch LBFGS).
        optimizer_kwargs: Dict of keyword arguments passed to the optimizer
            constructor.
        verbose: Print timing information.

    Returns:
        A new PoseStack with optimized coordinates.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    optimizer = optimizer_cls(sfxn_module.parameters(), **optimizer_kwargs)

    def closure():
        optimizer.zero_grad()
        E = sfxn_module().sum()
        E.backward()
        return E

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()
    optimizer.step(closure)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    new_pose_stack = sfxn_module.pose_stack_from_dofs()
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"run_min {end_time3 - start_time: .2f} setup: {end_time1-start_time: .2f}"
            + f" opt {end_time2 - end_time1: .2f} stack-ctor: {end_time3-end_time2: .2f}"
        )

    return new_pose_stack


def run_kin_min(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    optimizer_cls=LBFGS_Armijo,
    optimizer_kwargs=None,
    verbose=False,
    kin_dtype=torch.float32,
):
    """Run minimization on a PoseStack in internal DOF space.

    Builds a KinForestSfxnNetwork and delegates to run_min().
    """
    kf_network = build_kinforest_network(
        pose_stack, sfxn, ff, mm, verbose, kin_dtype=kin_dtype
    )
    return run_min(
        kf_network,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
    )


def run_cart_min(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    coord_mask=None,
    optimizer_cls=LBFGS_Armijo,
    optimizer_kwargs=None,
    verbose=False,
):
    """Run minimization on a PoseStack in Cartesian coordinate space.

    Builds a CartesianSfxnNetwork and delegates to run_min().
    """
    from tmol.optimization.sfxn_modules import CartesianSfxnNetwork

    cart_network = CartesianSfxnNetwork(sfxn, pose_stack, coord_mask)

    return run_min(
        cart_network,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
    )
