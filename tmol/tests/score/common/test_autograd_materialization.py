"""Unused native neighbor outputs must not allocate backward gradients."""

import pytest
import torch

from tmol import pose_stack_from_pdb
from tmol.score.lk_ball import LKBallEnergyTerm


def test_lk_ball_backward_does_not_materialize_neighbor_gradient(
    ubq_pdb, default_database, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("CUDA allocator statistics required")
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    term = LKBallEnergyTerm(param_db=default_database, device=torch_device)
    for block_type in pose.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    scorer = term.render_whole_pose_scoring_module(pose)
    coords = pose.coords.detach().requires_grad_(True)
    neighbors = scorer.build_compact_block_neighbors(
        coords, scorer.block_neighbor_cutoff
    )
    # Only the counted prefix participates in scoring.
    oversized = torch.full((1 << 20,), -1, dtype=torch.int32, device=torch_device)
    oversized[: neighbors.numel()].copy_(neighbors)
    expected = scorer(coords, neighbors)
    (expected_grad,) = torch.autograd.grad(expected.sum(), coords)
    for _ in range(2):
        warm = scorer(coords, oversized)
        torch.autograd.grad(warm.sum(), coords)
    actual = scorer(coords, oversized)
    torch.cuda.synchronize(torch_device)
    before = torch.cuda.memory_allocated(torch_device)
    torch.cuda.reset_peak_memory_stats(torch_device)
    (actual_grad,) = torch.autograd.grad(actual.sum(), coords)
    torch.cuda.synchronize(torch_device)
    peak = torch.cuda.max_memory_allocated(torch_device) - before
    assert peak < oversized.numel() * oversized.element_size()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
