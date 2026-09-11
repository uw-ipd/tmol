"""Task budgets reach reused samplers without changing caller-owned objects."""

import pytest

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import NaChiRotamerSampler, OptHSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.tests.pack.rotamer.test_na_chi_sampler import _pose_stack


@pytest.mark.parametrize("kind", ["opth", "na", "dunbrack"])
@pytest.mark.parametrize("before_adding_sampler", [False, True])
def test_task_limits_apply_when_a_sampler_is_reused(
    kind, before_adding_sampler, ubq_pdb, rna_pdb, default_database, torch_device
):
    pose = (
        _pose_stack(rna_pdb, torch_device)
        if kind == "na"
        else pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    )
    if kind == "opth":
        sampler = OptHSampler(flip_NHQ=False)
    elif kind == "na":
        sampler = NaChiRotamerSampler.from_database(default_database, torch_device)
    else:
        sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    for bt in pose.packed_block_types.active_block_types:
        sampler.annotate_residue_type(bt)
    sampler.annotate_packed_block_types(pose.packed_block_types)

    def sample(limit):
        task = PackerTask(pose, PackerPalette())
        task.restrict_to_repacking()
        if before_adding_sampler:
            task.set_chi_sample_budget(limit, limit)
        task.add_conformer_sampler(sampler)
        if not before_adding_sampler:
            task.set_chi_sample_budget(limit, limit)
        concrete = SetPackerTask.from_packer_task(task)
        task.set_chi_sample_budget(3000, 3000)
        assert concrete.chi_sample_budget == (limit, limit)
        assert (
            task.conformer_samplers[task.conformer_sampler_index[id(sampler)]]
            is sampler
        )
        if kind == "opth":
            return sampler.create_samples_for_poses(pose, concrete)[0]
        return sampler.sample_chi_for_poses(pose, concrete)[0]

    before = sample(3000)
    if kind == "dunbrack":
        # A budget cannot discard library states silently. The native count
        # check runs before rotamer mappings and chi tensors are allocated.
        with pytest.raises(RuntimeError, match="Sampling budget"):
            sample(1)
    else:
        limit = 2 if kind == "na" else 1
        bounded = sample(limit)
        assert int(bounded.max()) <= limit
        assert bool((before > bounded).any())
        assert sampler.chi_sample_limit == 1000
    after = sample(3000)
    assert bool((before == after).all())


def test_budget_setter_rejects_noninteger_limits_without_partial_update(ubq_pdb):
    import torch

    pose = pose_stack_from_pdb(
        ubq_pdb, torch.device("cpu"), residue_start=0, residue_end=2
    )
    task = PackerTask(pose, PackerPalette())
    task.set_chi_sample_budget(10, 20)
    for bad in (True, 2.5, 0, -1, "10"):
        with pytest.raises(ValueError, match="positive integers"):
            task.set_chi_sample_budget(bad, 20)
        assert task.chi_sample_budget == (10, 20)
