"""A wrapped count sum cannot validate an empty source mapping."""

import pytest
import torch

from tmol.pack.rotamer._build_rotamers import merge_conformer_samples


def test_int64_count_sum_cannot_wrap_into_an_empty_plan(torch_device):
    sample = (
        torch.full((4,), 2**62, dtype=torch.int64, device=torch_device),
        torch.empty(0, dtype=torch.int32, device=torch_device),
        {},
    )
    with pytest.raises(ValueError, match="sample counts"):
        merge_conformer_samples([sample])
