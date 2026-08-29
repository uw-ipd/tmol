from types import SimpleNamespace

import pytest
import torch
import biotite.structure as struc

from tmol.ops import build_coord_mask_for_mask_and_nearby_blocks
from tmol.io import (
    pose_stack_from_biotite,
    pose_stack_from_pdb,
    biotite_from_pose_stack,
)
from tmol.pose import PoseStackBuilder
from tmol import run_cart_min, beta2016_score_function
import tmol.ops._score_utils as score_utils


def test_sidechain_mask_treats_missing_mainchain_as_all_sidechain(torch_device):
    """Polymer metadata may omit main-chain atoms for ligand-like blocks."""
    residue_type = SimpleNamespace(
        properties=SimpleNamespace(
            polymer=SimpleNamespace(mainchain_atoms=None),
        ),
        atom_to_idx={},
    )
    packed_block_types = SimpleNamespace(
        atom_is_real=torch.tensor([[True, True, False]], device=torch_device),
        active_block_types=[residue_type],
    )
    pose_stack = SimpleNamespace(packed_block_types=packed_block_types)

    actual = score_utils._sidechain_atom_mask_for_block_type(pose_stack)

    torch.testing.assert_close(actual, packed_block_types.atom_is_real)


@pytest.mark.parametrize("einsum", [False, True])
def test_memory_efficient_sum_handles_different_mask_sizes(
    torch_device, monkeypatch, einsum
):
    """Each pose may select a different number of block pairs."""
    monkeypatch.setattr(score_utils, "_EINSUM_MIN_BYTES", 0 if einsum else float("inf"))
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
    scores = torch.randn(5, 3, 32, 32, generator=generator, device=torch_device)
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


def test_single_block_indices_match_boolean_masks(torch_device):
    generator = torch.Generator(device=torch_device).manual_seed(29)
    scores = torch.randn(5, 3, 7, 7, generator=generator, device=torch_device)
    block_indices = torch.tensor([1, 4, 2], device=torch_device)
    mask = torch.zeros(3, 7, dtype=torch.bool, device=torch_device)
    mask.scatter_(1, block_indices[:, None], True)
    other = torch.tensor(
        [
            [True, True, False, True, False, False, True],
            [False, True, True, False, False, True, False],
            [True, False, False, True, True, False, True],
        ],
        device=torch_device,
    )

    expected = score_utils._sum_cross_block_scores(
        scores, mask, other, memory_efficient=True
    )
    actual = score_utils._sum_single_block_cross_scores(scores, block_indices, other)

    torch.testing.assert_close(actual, expected)


def test_calculate_ddg_accepts_single_block_indices(torch_device):
    class Pose:
        device = torch_device
        n_poses = 2
        block_type_ind = torch.zeros(2, 4, dtype=torch.int32, device=torch_device)
        coords = torch.empty(2, 0, 3, device=torch_device)

    scores = torch.arange(2 * 2 * 4 * 4, device=torch_device, dtype=torch.float32)
    scores = scores.reshape(2, 2, 4, 4)
    weights = torch.tensor([0.25, 1.5], device=torch_device).reshape(2, 1, 1, 1)

    class ScoreFunction:
        @staticmethod
        def render_block_pair_scoring_module(_pose):
            class Scorer:
                def __init__(self):
                    self.weights = weights

                def __call__(self, _coords, sum_terms, apply_weights=True):
                    return weights * scores if apply_weights else scores

            return Scorer()

    block_indices = torch.tensor([0, 2], device=torch_device)
    mask = torch.zeros(2, 4, dtype=torch.bool, device=torch_device)
    mask.scatter_(1, block_indices[:, None], True)
    expected = score_utils.calculate_block_pair_ddg(
        Pose(), mask, sfxn=ScoreFunction(), minimize=False
    )
    actual = score_utils.calculate_block_pair_ddg(
        Pose(), block_indices, sfxn=ScoreFunction(), minimize=False
    )

    torch.testing.assert_close(actual, expected)

    with pytest.raises(TypeError, match="integer dtype"):
        score_utils.calculate_block_pair_ddg(
            Pose(), block_indices.float(), sfxn=ScoreFunction(), minimize=False
        )


def test_interacting_coord_mask_matches_per_pose_reference(
    biotite_1ubq: struc.AtomArray, systems_bysize, torch_device
):
    pose = pose_stack_from_biotite(biotite_1ubq, torch_device)
    shorter_pose = pose_stack_from_pdb(systems_bysize[40], torch_device)
    poses = PoseStackBuilder.from_poses([pose, shorter_pose, pose], torch_device)
    block_mask = torch.zeros_like(poses.block_type_ind, dtype=torch.bool)
    block_mask[0, 0] = True
    block_mask[1, 5] = True
    block_mask[2, [0, 5]] = True

    residue_mask = score_utils.res_mask_to_coord_mask(poses, block_mask)
    sidechain_mask = score_utils.build_sidechain_coord_mask(poses)
    actual = score_utils.build_coord_mask_for_mask_and_interacting_atoms(
        poses, block_mask
    )

    expected_residue_mask = torch.zeros_like(residue_mask)
    expected_sidechain_mask = torch.zeros_like(sidechain_mask)
    pbt = poses.packed_block_types
    for pose_index in range(poses.n_poses):
        for block_index in range(poses.max_n_blocks):
            block_type_index = int(poses.block_type_ind64[pose_index, block_index])
            if block_type_index < 0:
                continue
            residue_type = pbt.active_block_types[block_type_index]
            block_offset = int(poses.block_coord_offset64[pose_index, block_index])
            n_atoms = int(pbt.n_atoms[block_type_index])
            if block_mask[pose_index, block_index]:
                expected_residue_mask[
                    pose_index, block_offset : block_offset + n_atoms
                ] = True

            mainchain_atoms = set(residue_type.properties.polymer.mainchain_atoms)
            for atom_index, atom in enumerate(residue_type.atoms):
                if atom.name not in mainchain_atoms:
                    expected_sidechain_mask[pose_index, block_offset + atom_index] = (
                        True
                    )

    assert torch.equal(residue_mask, expected_residue_mask)
    assert torch.equal(sidechain_mask, expected_sidechain_mask)
    assert score_utils.build_sidechain_coord_mask(poses) is sidechain_mask

    expected = residue_mask.clone()
    for pose_index in range(poses.n_poses):
        selected_coords = poses.coords[pose_index, residue_mask[pose_index]]
        differences = poses.coords[pose_index, :, None, :] - selected_coords[None, :, :]
        nearby = torch.sqrt((differences**2).sum(dim=2)).amin(dim=1) <= 5.0
        expected[pose_index] |= (
            nearby & poses.real_atoms[pose_index] & sidechain_mask[pose_index]
        )

    assert torch.equal(actual, expected)
    assert torch.equal(
        actual,
        score_utils.build_coord_mask_for_mask_and_interacting_atoms(poses, block_mask),
    )

    assert torch.equal(
        actual,
        score_utils.build_coord_mask_for_mask_and_interacting_atoms(
            poses, block_mask, max_workspace_bytes=1
        ),
    )
    assert not score_utils.build_coord_mask_for_mask_and_interacting_atoms(
        poses, torch.zeros_like(block_mask)
    ).any()
    assert torch.equal(
        score_utils.build_coord_mask_for_mask_and_interacting_atoms(
            poses, block_mask, interaction_distance=-1
        ),
        residue_mask,
    )
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        score_utils.build_coord_mask_for_mask_and_interacting_atoms(
            poses, block_mask, max_workspace_bytes=0
        )


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
