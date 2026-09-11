"""Reject malformed sampler plans before indexed writes into merged arrays."""

import pytest
import torch

from tmol.pack.rotamer._build_rotamers import merge_conformer_samples


@pytest.mark.parametrize(
    "counts,indices",
    [
        ([1, 1], [1, 0]),
        ([2, 1], [0, 1, 1]),
        ([1, 1], [-1, 1]),
        ([1, 1], [0, 2]),
        ([3, 2], [0, 1, 0, 1, 1]),
    ],
)
def test_source_rows_must_match_ordered_count_intervals(counts, indices, torch_device):
    sample = (
        torch.tensor(counts, dtype=torch.int32, device=torch_device),
        torch.tensor(indices, dtype=torch.int32, device=torch_device),
        {},
    )
    with pytest.raises(ValueError, match="source rows.*ordered sample counts"):
        merge_conformer_samples([sample])
