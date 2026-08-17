import time
import torch
from tmol.pose import PoseStack

from tmol.kinematics import FoldForest
from tmol.kinematics import MoveMap, MinimizerMap
from tmol.score import ScoreFunction

from tmol.optimization import LBFGS_Armijo


def build_kinforest_network(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose=False,
    kin_dtype=torch.float32,
):
    from tmol.kinematics import PoseStackKinematicsModule
    from tmol.optimization import KinForestSfxnNetwork

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
            + f" s1: {end_time1 - start_time: .2f} s2: {end_time2 - end_time1: .2f}"
            + f" s3: {end_time3 - end_time2: .2f}"
        )

    return kf_network


def run_min(
    sfxn_module,
    optimizer_cls=LBFGS_Armijo,
    optimizer_kwargs=None,
    verbose=False,
    per_pose=True,
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
        per_pose: Give each pose its own inverse-Hessian estimate and
            convergence test, so that minimizing a stack matches minimizing
            its poses one at a time. Ignored by optimizers that do not
            support it.

    Returns:
        A new PoseStack with optimized coordinates.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    segment_ids = getattr(sfxn_module, "segment_ids", None)
    segmented = (
        per_pose
        and segment_ids is not None
        and getattr(optimizer_cls, "supports_segments", False)
    )
    if segmented:
        optimizer_kwargs = dict(optimizer_kwargs, segment_ids=segment_ids)

    optimizer = optimizer_cls(sfxn_module.parameters(), **optimizer_kwargs)

    def closure():
        optimizer.zero_grad()
        E = sfxn_module()
        E.sum().backward()
        return E if segmented else E.sum()

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
            f"run_min {end_time3 - start_time: .2f} setup: {end_time1 - start_time: .2f}"
            + f" opt {end_time2 - end_time1: .2f} stack-ctor: {end_time3 - end_time2: .2f}"
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
    from tmol.optimization import CartesianSfxnNetwork

    cart_network = CartesianSfxnNetwork(sfxn, pose_stack, coord_mask)

    return run_min(
        cart_network,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
    )
