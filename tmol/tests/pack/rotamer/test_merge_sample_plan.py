"""A merged row preserves its sampler, considered type and original row index."""

import pytest
import torch

from tmol.pack.rotamer._build_rotamers import merge_conformer_samples


def samples_from_counts(rows, device):
    return [
        (
            torch.tensor(row, dtype=torch.int32, device=device),
            torch.repeat_interleave(
                torch.arange(len(row), dtype=torch.int32, device=device),
                torch.tensor(row, device=device, dtype=torch.int64),
            ),
            {},
        )
        for row in rows
    ]


@pytest.mark.parametrize(
    "counts",
    [
        [[2, 0, 1, 0], [0, 3, 2, 0], [1, 0, 0, 0]],
        [[0, 0], [0, 0]],
        [[], []],
        [[3, 1, 0, 2]],
        [[0, 0, 0], [2, 1, 3], [0, 0, 0]],
    ],
)
@pytest.mark.parametrize("mixed_indices", [False, True])
def test_merge_preserves_source_rows_and_empty_slots(
    counts, torch_device, mixed_indices
):
    samples = samples_from_counts(counts, torch_device)
    if mixed_indices:
        counts0, indices, extra = samples[0]
        samples[0] = (counts0, indices.long(), extra)
    merged_counts, sampler_ids, gbts, masks, destinations = merge_conformer_samples(
        samples
    )
    # Independent record enumeration, ordered by type, then sampler, then row.
    records = [
        (gbt, sampler, sum(counts[sampler][:gbt]) + local)
        for gbt in range(len(counts[0]))
        for sampler in range(len(counts))
        for local in range(counts[sampler][gbt])
    ]
    assert merged_counts.tolist() == [
        sum(row[g] for row in counts) for g in range(len(counts[0]))
    ]
    assert gbts.tolist() == [r[0] for r in records]
    assert sampler_ids.tolist() == [r[1] for r in records]
    for i, (mask, destination) in enumerate(zip(masks, destinations)):
        assert mask.tolist() == [r[1] == i for r in records]
        expected = [k for k, r in enumerate(records) if r[1] == i]
        assert destination.tolist() == expected
        # Payloads with distinct source row IDs test the inverse mapping.
        payload = torch.full(
            (len(records),), -1, dtype=torch.int64, device=torch_device
        )
        payload[destination] = torch.arange(len(destination), device=torch_device)
        assert payload[mask].tolist() == [r[2] for r in records if r[1] == i]


@pytest.mark.parametrize("counts", [[-1, 2], [1, 2]])
def test_merge_rejects_invalid_counts_before_scattering(counts, torch_device):
    sample = (
        torch.tensor(counts, dtype=torch.int32, device=torch_device),
        torch.tensor([1], dtype=torch.int32, device=torch_device),
        {},
    )
    with pytest.raises(ValueError, match="sample counts"):
        merge_conformer_samples([sample])
