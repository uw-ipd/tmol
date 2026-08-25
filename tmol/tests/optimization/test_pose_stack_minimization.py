"""Minimizing a stack of distinct poses should match minimizing them one-by-one."""

import pytest
import torch

from tmol import (
    PoseStack,
    beta2016_score_function,
    run_cart_min,
    run_kin_min,
    FoldForest,
    MoveMap,
)
from tmol.pose import PoseStackBuilder
from tmol.optimization import CartesianSfxnNetwork, KinForestSfxnNetwork
from tmol.kinematics import PoseStackKinematicsModule


def _score_per_pose(pose_stack: PoseStack, sfxn):
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    return wpsm(pose_stack.coords).detach()


def _cart_min_per_pose(pose_stack: PoseStack, sfxn):
    return _score_per_pose(run_cart_min(pose_stack, sfxn), sfxn)


def _kin_min_per_pose(pose_stack: PoseStack, sfxn):
    ff = FoldForest.reasonable_fold_forest(pose_stack)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    return _score_per_pose(run_kin_min(pose_stack, sfxn, ff, mm), sfxn)


def _report(label, start, one_by_one, stacked):
    lines = [
        f"{label}: per-pose energies",
        f"{'pose':>4} {'start':>12} {'one-by-one':>12} {'stacked':>12} {'delta':>10}",
    ]
    for i in range(len(stacked)):
        lines.append(
            f"{i:>4} {start[i]:>12.3f} {one_by_one[i]:>12.3f}"
            f" {stacked[i]:>12.3f} {stacked[i] - one_by_one[i]:>10.3f}"
        )
    return "\n".join(lines)


def _compare(label, poses, stack, sfxn, min_per_pose, tol=5.0):
    start = _score_per_pose(stack, sfxn)
    one_by_one = torch.cat([min_per_pose(p, sfxn) for p in poses])
    stacked = min_per_pose(stack, sfxn)

    report = _report(label, start, one_by_one, stacked)
    assert torch.all(stacked < start), report
    # FP32 reduction order differs between a heterogeneous stack and separate
    # runs, so nonlinear line-search trajectories need a loose energy bound.
    assert torch.all(torch.abs(stacked - one_by_one) < tol), report


def test_score_stack_of_distinct_poses_matches_individual(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Baseline: scoring is stack-invariant, so any difference is the minimizer's."""
    sfxn = beta2016_score_function(torch_device)
    individual = torch.cat([_score_per_pose(p, sfxn) for p in distinct_pose_stacks])
    stacked = _score_per_pose(stack_of_distinct_poses, sfxn)
    torch.testing.assert_close(stacked, individual, rtol=1e-4, atol=1e-3)


def test_cart_min_stack_of_distinct_poses(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    sfxn = beta2016_score_function(torch_device)
    _compare(
        "cart, distinct poses",
        distinct_pose_stacks,
        stack_of_distinct_poses,
        sfxn,
        _cart_min_per_pose,
    )


@pytest.mark.xfail
def test_kin_min_stack_of_distinct_poses(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    sfxn = beta2016_score_function(torch_device)
    _compare(
        "kin, distinct poses",
        distinct_pose_stacks,
        stack_of_distinct_poses,
        sfxn,
        _kin_min_per_pose,
    )


def test_cart_network_segment_ids(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Each pose's coordinate DOFs must be labelled with that pose."""
    sfxn = beta2016_score_function(torch_device)
    network = CartesianSfxnNetwork(sfxn, stack_of_distinct_poses)

    segment_ids = network.segment_ids
    assert segment_ids.shape == (network.masked_coords.numel(),)
    counts = torch.bincount(segment_ids)
    assert len(counts) == len(distinct_pose_stacks)
    solo_counts = torch.tensor(
        [
            CartesianSfxnNetwork(sfxn, pose).masked_coords.numel()
            for pose in distinct_pose_stacks
        ],
        device=counts.device,
    )
    # Padding in a heterogeneous stack must not create optimizer variables.
    assert torch.all(counts == solo_counts), f"{counts} != {solo_counts}"
    # coords are laid out pose-major, so the labels come in contiguous runs
    assert torch.all(segment_ids[1:] >= segment_ids[:-1])


def test_kin_network_segment_ids(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Each pose's torsion DOFs must be labelled with that pose."""
    sfxn = beta2016_score_function(torch_device)

    def network_for(pose_stack):
        kin_module = PoseStackKinematicsModule(
            pose_stack, FoldForest.reasonable_fold_forest(pose_stack)
        )
        return KinForestSfxnNetwork(sfxn, pose_stack, kin_module)

    network = network_for(stack_of_distinct_poses)
    segment_ids = network.segment_ids
    assert segment_ids.shape == (network.masked_dofs.numel(),)

    counts = torch.bincount(segment_ids, minlength=len(distinct_pose_stacks))
    solo_counts = torch.tensor(
        [network_for(p).masked_dofs.numel() for p in distinct_pose_stacks],
        device=counts.device,
    )
    # a pose contributes the same dofs whether it is minimized alone or in a stack
    assert torch.all(counts == solo_counts), f"{counts} != {solo_counts}"


def test_cart_min_stack_of_identical_poses(distinct_pose_stacks, torch_device):
    """Control: a stack of copies of one pose should match minimizing it alone."""
    sfxn = beta2016_score_function(torch_device)
    poses = [distinct_pose_stacks[0]] * 3
    stack = PoseStackBuilder.from_poses(poses, torch_device)
    _compare("cart, identical poses", poses, stack, sfxn, _cart_min_per_pose)
