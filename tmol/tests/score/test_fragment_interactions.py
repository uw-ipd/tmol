from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tmol.pose import PoseStack, SplitBlockMapping
from tmol.score._fragment_interactions import (
    _LegacyFragmentMapping,
    _fragment_mapping_data,
    _sum_fragment_partner_scores,
)


def _pose_topology(torch_device: torch.device) -> PoseStack:
    return cast(
        PoseStack,
        SimpleNamespace(
            n_poses=2,
            max_n_blocks=4,
            block_type_ind=torch.zeros((2, 4), dtype=torch.int32, device=torch_device),
        ),
    )


def _legacy_mapping(*records: tuple[int, int, int]) -> _LegacyFragmentMapping:
    return cast(
        _LegacyFragmentMapping,
        SimpleNamespace(
            blocks=tuple(
                SimpleNamespace(
                    pose_index=pose_index,
                    block_index=block_index,
                    pose_residue_label=residue_label,
                )
                for pose_index, block_index, residue_label in records
            )
        ),
    )


def test_fragment_mapping_normalizes_legacy_records(
    torch_device: torch.device,
) -> None:
    pose_stack = _pose_topology(torch_device)
    mapping = _legacy_mapping(
        (0, 3, 8),
        (0, 1, 7),
        (1, 3, 8),
        (1, 1, 7),
    )

    data = _fragment_mapping_data(pose_stack, mapping)

    assert [record.block_index for record in data.fragment_records] == [1, 3]
    torch.testing.assert_close(
        data.record_linear_indices,
        torch.tensor([3, 1, 7, 5], dtype=torch.int64, device=torch_device),
    )
    assert _fragment_mapping_data(pose_stack, mapping) is data


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (SplitBlockMapping(entries=()), "no entries for pose 0"),
        (
            _legacy_mapping((0, 1, 7), (1, 2, 7)),
            "identical block topology",
        ),
        (
            _legacy_mapping((0, 4, 7), (1, 4, 7)),
            "outside the pose stack",
        ),
        (
            _legacy_mapping((0, 1, 7), (0, 1, 7), (1, 1, 7)),
            "duplicate columns",
        ),
    ],
)
def test_fragment_mapping_rejects_invalid_topology(
    torch_device: torch.device,
    mapping: SplitBlockMapping | _LegacyFragmentMapping,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _fragment_mapping_data(_pose_topology(torch_device), mapping)


def _reference_fragment_partner_scores(
    block_pair_scores: torch.Tensor,
    partner_mask: torch.Tensor,
    fragment_block_indices: torch.Tensor,
) -> torch.Tensor:
    partner = partner_mask.unsqueeze(0)
    return torch.stack(
        [
            (block_pair_scores[:, :, block_index, :] * partner).sum(dim=2)
            + (block_pair_scores[:, :, :, block_index] * partner).sum(dim=2)
            for block_index in fragment_block_indices.tolist()
        ],
        dim=2,
    )


@pytest.mark.parametrize("n_fragments", [1, 5])
def test_fragment_partner_reduction_matches_loop(
    torch_device: torch.device, n_fragments: int
) -> None:
    generator = torch.Generator(device=torch_device).manual_seed(17)
    scores = torch.randn(
        (3, 2, 9, 9),
        generator=generator,
        device=torch_device,
        requires_grad=True,
    )
    partner_mask = torch.rand((2, 9), generator=generator, device=torch_device) > 0.3
    fragment_indices = torch.arange(n_fragments, device=torch_device)

    expected = _reference_fragment_partner_scores(
        scores, partner_mask, fragment_indices
    )
    bytes_per_fragment = 2 * 3 * 2 * 9 * scores.element_size()
    actual = _sum_fragment_partner_scores(
        scores,
        partner_mask,
        fragment_indices,
        max_workspace_bytes=2 * bytes_per_fragment,
    )
    torch.testing.assert_close(actual, expected)

    expected_gradient = torch.autograd.grad(expected.sum(), scores, retain_graph=True)[
        0
    ]
    actual_gradient = torch.autograd.grad(actual.sum(), scores)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_fragment_partner_reduction_accepts_no_fragments(
    torch_device: torch.device,
) -> None:
    scores = torch.zeros((3, 2, 9, 9), device=torch_device, requires_grad=True)
    partner_mask = torch.zeros((2, 9), dtype=torch.bool, device=torch_device)
    fragment_indices = torch.empty((0,), dtype=torch.int64, device=torch_device)

    result = _sum_fragment_partner_scores(scores, partner_mask, fragment_indices)

    assert result.shape == (3, 2, 0)
    assert result.dtype == scores.dtype
    assert result.device == scores.device
    assert result.requires_grad


@pytest.mark.benchmark(group="fragment_partner_reduction")
def test_fragment_partner_reduction_benchmark(
    benchmark, torch_device: torch.device
) -> None:
    generator = torch.Generator(device=torch_device).manual_seed(17)
    scores = torch.randn((22, 8, 100, 100), generator=generator, device=torch_device)
    partner_mask = torch.rand((8, 100), generator=generator, device=torch_device) > 0.2
    fragment_indices = torch.arange(7, device=torch_device)

    def reduce_fragment_scores() -> torch.Tensor:
        result = _sum_fragment_partner_scores(scores, partner_mask, fragment_indices)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        return result

    reduce_fragment_scores()  # Warm up the CUDA reduction before timing.
    benchmark(reduce_fragment_scores)
