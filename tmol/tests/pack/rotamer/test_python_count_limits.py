"""Sampler allocation bounds must be checked before count narrowing."""

import pytest
import torch

from tmol.pack.rotamer._chi_budget import checked_sample_count

INT_MAX = torch.iinfo(torch.int32).max


@pytest.mark.parametrize(
    "values",
    [
        [-1, 2],
        [INT_MAX + 1],
        [INT_MAX, 1],
        [2**62] * 4,  # an int64 sum can wrap back to zero
    ],
)
def test_invalid_sample_counts_cannot_size_allocations(values, torch_device):
    counts = torch.tensor(values, dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="Sampling count"):
        checked_sample_count(counts, 2**63, 2**63)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("values", [[], [0], [0, 2, 3], [INT_MAX - 1, 1]])
def test_valid_sample_count_boundaries(values, dtype, torch_device):
    counts = torch.tensor(values, dtype=dtype, device=torch_device)
    assert checked_sample_count(counts, INT_MAX, INT_MAX) == sum(values)


def test_per_type_budget_remains_separate_from_total(torch_device):
    counts = torch.tensor([3, 3], device=torch_device)
    assert checked_sample_count(counts, 2, 3) == 6
    with pytest.raises(ValueError, match="Sampling budget 2"):
        checked_sample_count(counts, 2, 2)


@pytest.mark.parametrize("combos", [2**32, 2**31, 2**62, -1])
def test_nucleotide_counts_checked_before_cast_and_row_allocation(
    combos, rna_pdb, default_database, torch_device, monkeypatch
):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import NaChiRotamerSampler
    from tmol.tests.pack.rotamer.test_na_chi_sampler import _pose_stack

    pose = _pose_stack(rna_pdb, torch_device)
    sampler = NaChiRotamerSampler.from_database(
        default_database, torch_device, chi_sample_level=1
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    concrete = SetPackerTask.from_packer_task(task)
    cache = dict(sampler.annotate_packed_block_types(pose.packed_block_types))
    cache["n_combos"] = torch.full_like(cache["n_combos"], combos)
    monkeypatch.setattr(
        NaChiRotamerSampler, "annotate_packed_block_types", lambda self, pbt: cache
    )

    def premature_allocation(*args, **kwargs):
        raise AssertionError("Invalid counts must fail before rotamer row allocation")

    monkeypatch.setattr(torch, "repeat_interleave", premature_allocation)
    with pytest.raises(ValueError, match="Sampling count"):
        sampler.sample_chi_for_poses(pose, concrete)
