import pytest
import torch
import biotite.structure as struc

from tmol.ops import build_coord_mask_for_mask_and_nearby_blocks
from tmol.io import (
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)
from tmol import run_cart_min, beta2016_score_function
import tmol.ops._score_utils as score_utils


@pytest.mark.parametrize("einsum", [False, True])
def test_memory_efficient_sum_handles_different_mask_sizes(
    torch_device, monkeypatch, einsum
):
    """Each pose may select a different number of block pairs."""
    monkeypatch.setattr(
        score_utils, "_EINSUM_MIN_BYTES", 0 if einsum else float("inf")
    )
    scores = torch.arange(
        2 * 2 * 4 * 4, dtype=torch.float32, device=torch_device
    ).reshape(2, 2, 4, 4)
    scores.requires_grad_(True)
    mask = torch.tensor(
        [[True, False, False, False], [True, True, False, False]],
        device=torch_device,
    )
    other = torch.tensor(
        [[False, True, False, False], [False, False, True, True]],
        device=torch_device,
    )

    actual = score_utils._sum_cross_block_scores(
        scores, mask, other, memory_efficient=True
    )
    expected = torch.empty_like(actual)
    expected[:, 0] = scores[:, 0, 0, 1] + scores[:, 0, 1, 0]
    expected[:, 1] = scores[:, 1, :2, 2:].sum(dim=(1, 2)) + scores[:, 1, 2:, :2].sum(
        dim=(1, 2)
    )
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    selected = (mask.unsqueeze(2) & other.unsqueeze(1)) | (
        other.unsqueeze(2) & mask.unsqueeze(1)
    )
    torch.testing.assert_close(
        scores.grad,
        selected.unsqueeze(0).expand_as(scores).to(scores.dtype),
    )


def test_default_sum_is_bitwise_legacy_equivalent(torch_device):
    """The default retains the legacy selection and sum order."""
    generator = torch.Generator(device=torch_device).manual_seed(17)
    scores = torch.randn(
        5, 3, 32, 32, generator=generator, device=torch_device
    )
    mask = torch.zeros(3, 32, dtype=torch.bool, device=torch_device)
    mask[:, :11] = True
    other = ~mask
    cross_mask = (mask.unsqueeze(2) & other.unsqueeze(1)) | (
        other.unsqueeze(2) & mask.unsqueeze(1)
    )

    expanded_mask = cross_mask.unsqueeze(0).expand(scores.shape[0], -1, -1, -1)
    expected = scores[expanded_mask].view(5, 3, -1).sum(dim=2)
    actual = score_utils._sum_cross_block_scores(scores, mask, other)

    assert torch.equal(actual, expected)


def test_build_coord_mask_and_minimize_for_first_residue(
    biotite_1ubq: struc.AtomArray, torch_device
):
    """Load 1ubq from biotite fixture, mask first residue, build coord_mask,
    run cart_min on the masked atoms, and convert back to biotite."""

    # Convert biotite structure to a PoseStack
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device)

    # Build a mask: True for the first residue (block 0)
    mask = torch.zeros_like(
        pose_stack.block_coord_offset, dtype=torch.bool, device=torch_device
    )
    mask[:, 0] = True

    # Generate the coord_mask from the block mask
    coord_mask = build_coord_mask_for_mask_and_nearby_blocks(pose_stack, mask)

    # Verify the coord_mask has been produced
    assert coord_mask.shape == pose_stack.coords.shape[:2], (
        f"coord_mask shape {coord_mask.shape} does not match "
        f"expected {pose_stack.coords.shape[:2]}"
    )
    assert coord_mask.dtype == torch.bool
    n_true = coord_mask.count_nonzero().item()
    assert (
        n_true > 0
    ), "coord_mask should have at least one True entry for the first residue"

    # Run cartesian minimization with the coord_mask
    sfxn = beta2016_score_function(torch_device)
    minimized_pose = run_cart_min(pose_stack, sfxn, coord_mask)

    # Convert the minimized pose back to a biotite structure
    result_biotite = biotite_from_pose_stack(minimized_pose)

    # Check that the output is a valid biotite AtomArray
    assert isinstance(
        result_biotite, struc.AtomArray
    ), f"Expected biotite AtomArray, got {type(result_biotite)}"
    assert result_biotite.array_length() > 0, "Resulting biotite structure is empty"
    assert not torch.any(
        torch.isnan(minimized_pose.coords)
    ), "Minimized pose contains NaN coordinates"
