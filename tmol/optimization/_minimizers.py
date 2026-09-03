from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from tmol.pose import PoseStack
from tmol.types import Tensor

from tmol.kinematics import (
    FoldForest,
    MoveMap,
    MinimizerMap,
)
from tmol.score import ScoreFunction

from tmol.optimization import LBFGS_Armijo
from tmol.utility._device import synchronize_device

if TYPE_CHECKING:
    from tmol.optimization import CartesianSfxnNetwork, KinForestSfxnNetwork


def build_kinforest_network(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    ff: FoldForest,
    mm: MoveMap,
    verbose: bool = False,
    kin_dtype: torch.dtype = torch.float32,
) -> KinForestSfxnNetwork:
    """Build a differentiable kinematic scoring network for a pose stack.

    Args:
        pose_stack: Structures and topology to score.
        sfxn: Score function to render against ``pose_stack``.
        ff: Fold forest defining the kinematic tree.
        mm: Internal-coordinate degrees of freedom allowed to move.
        verbose: Print synchronized setup timings.
        kin_dtype: Floating-point dtype for kinematic degrees of freedom.

    Returns:
        A network mapping movable kinematic degrees of freedom to pose energies.
    """
    from tmol.kinematics import PoseStackKinematicsModule
    from tmol.optimization import KinForestSfxnNetwork

    if verbose:
        synchronize_device(pose_stack.device)
    start_time = time.perf_counter()

    kin_module = PoseStackKinematicsModule(pose_stack, ff)
    if verbose:
        synchronize_device(pose_stack.device)
    end_time1 = time.perf_counter()

    minimizer_map = MinimizerMap(pose_stack, kin_module.kmd, mm)
    if verbose:
        synchronize_device(pose_stack.device)
    end_time2 = time.perf_counter()

    kf_network = KinForestSfxnNetwork(
        sfxn, pose_stack, kin_module, minimizer_map.dof_mask, kin_dtype=kin_dtype
    )
    if verbose:
        synchronize_device(pose_stack.device)
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"build_kinforest_network {end_time3 - start_time: .2f}"
            + f" s1: {end_time1 - start_time: .2f} s2: {end_time2 - end_time1: .2f}"
            + f" s3: {end_time3 - end_time2: .2f}"
        )

    return kf_network


def run_min(
    sfxn_module: CartesianSfxnNetwork | KinForestSfxnNetwork,
    optimizer_cls: type[torch.optim.Optimizer] = LBFGS_Armijo,
    optimizer_kwargs: dict[str, object] | None = None,
    verbose: bool = False,
    per_pose: bool = True,
) -> PoseStack:
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

    timing_device = next(sfxn_module.parameters()).device if verbose else None
    if timing_device is not None:
        synchronize_device(timing_device)
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

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        E = sfxn_module()
        total = E.sum()
        total.backward()
        return E if segmented else total

    if timing_device is not None:
        synchronize_device(timing_device)
    end_time1 = time.perf_counter()
    optimizer.step(closure)
    if timing_device is not None:
        synchronize_device(timing_device)
    end_time2 = time.perf_counter()

    new_pose_stack = sfxn_module.pose_stack_from_dofs()
    if timing_device is not None:
        synchronize_device(timing_device)
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
    optimizer_cls: type[torch.optim.Optimizer] = LBFGS_Armijo,
    optimizer_kwargs: dict[str, object] | None = None,
    verbose: bool = False,
    kin_dtype: torch.dtype = torch.float32,
) -> PoseStack:
    """Run minimization on a PoseStack in internal DOF space.

    Builds a ``KinForestSfxnNetwork`` and delegates to :func:`run_min`.

    Args:
        pose_stack: Structures and coordinates to minimize.
        sfxn: Score function defining the objective.
        ff: Fold forest defining the kinematic tree.
        mm: Internal-coordinate degrees of freedom allowed to move.
        optimizer_cls: Closure-based PyTorch optimizer class.
        optimizer_kwargs: Optional optimizer constructor arguments.
        verbose: Print synchronized setup and minimization timings.
        kin_dtype: Floating-point dtype for kinematic degrees of freedom.

    Returns:
        A new pose stack containing the minimized coordinates.
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
    coord_mask: Tensor[torch.bool][:, :] | None = None,
    optimizer_cls: type[torch.optim.Optimizer] = LBFGS_Armijo,
    optimizer_kwargs: dict[str, object] | None = None,
    verbose: bool = False,
    cuda_graph: bool = False,
) -> PoseStack:
    """Run minimization on a PoseStack in Cartesian coordinate space.

    Builds a ``CartesianSfxnNetwork`` and delegates to :func:`run_min`.

    Args:
        pose_stack: Structures and coordinates to minimize.
        sfxn: Score function defining the objective.
        coord_mask: Movable atoms shaped ``[pose, atom]``; ``None`` moves all.
        optimizer_cls: Closure-based PyTorch optimizer class.
        optimizer_kwargs: Optional optimizer constructor arguments.
        verbose: Print synchronized setup and minimization timings.
        cuda_graph: Capture the fixed-shape CUDA scoring forward/backward path.

    Returns:
        A new pose stack containing the minimized coordinates.
    """
    from tmol.optimization import CartesianSfxnNetwork

    cart_network = CartesianSfxnNetwork(
        sfxn,
        pose_stack,
        coord_mask,
        cuda_graph="forward_backward" if cuda_graph else False,
    )

    return run_min(
        cart_network,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        verbose=verbose,
    )
