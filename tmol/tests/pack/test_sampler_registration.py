import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler
from tmol.pose import PoseStackBuilder


@pytest.mark.parametrize(
    "registration", ["split", "repeat", "all", "defaults", "distinct"]
)
def test_registration_samples_each_enabled_residue_once(
    ubq_pdb, torch_device, monkeypatch, registration
):
    poses = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=3),
            pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5),
        ],
        torch_device,
    )
    sampler = IncludeCurrentSampler()
    defaults = [sampler, sampler] if registration == "defaults" else []
    monkeypatch.setattr(
        PackerPalette, "default_conformer_samplers", lambda self: defaults
    )
    task = PackerTask(poses, PackerPalette())
    task.restrict_to_repacking()
    first = torch.zeros_like(task.is_real_block)
    first[0, :2] = True
    second = torch.zeros_like(first)
    second[1, 1:4] = True
    expected = first | second
    if registration == "defaults":
        expected = task.is_real_block
    else:
        task.add_conformer_sampler_by_block_mask(sampler, first)
        if registration == "all":
            task.add_conformer_sampler(sampler)
            expected = task.is_real_block
        elif registration == "distinct":
            other = IncludeCurrentSampler()
            assert other == sampler and other is not sampler
            task.add_conformer_sampler_by_block_mask(other, second)
        else:
            task.add_conformer_sampler_by_block_mask(sampler, second)
            if registration == "repeat":
                task.add_conformer_sampler_by_block_mask(sampler, second)

    finalized = SetPackerTask.from_packer_task(task)
    counts = torch.zeros_like(first, dtype=torch.int64)
    for current in finalized.conformer_samplers:
        _, rows, _ = current.create_samples_for_poses(poses, finalized)
        counts.index_put_(
            (finalized.cons_bt_pose[rows], finalized.cons_bt_block[rows]),
            torch.ones_like(rows),
            accumulate=True,
        )
    torch.testing.assert_close(counts, expected.to(torch.int64))
    assert len(task.conformer_samplers) == (2 if registration == "distinct" else 1)
    # Tasks own their registration list even when a palette reuses its list.
    assert defaults == ([sampler, sampler] if registration == "defaults" else [])

    if registration != "distinct":
        storage = task.per_block_conformer_sampler_allowed.data_ptr()
        task.disable_sampler_by_block_mask(sampler, first)
        task.add_conformer_sampler_by_block_mask(sampler, first)
        task.add_conformer_sampler(sampler)
        assert task.per_block_conformer_sampler_allowed.data_ptr() == storage
        assert len(task.conformer_samplers) == 1
        assert task.per_block_conformer_sampler_allowed.all()
