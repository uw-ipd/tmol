import torch
import biotite.structure as struc

from tmol.ops import build_coord_mask_for_mask_and_nearby_blocks
from tmol.io import (
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)
from tmol import run_cart_min, beta2016_score_function


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
