"""Shared chemical types must use the calling sampler's library mapping."""

import copy

import attr
import numpy
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer.dunbrack import DunbrackChiSampler
from tmol.pose import PackedBlockTypes
from tmol.score.dunbrack import DunbrackParamResolver


def fresh_pose(pose):
    old = pose.packed_block_types
    types = [copy.copy(bt) for bt in old.active_block_types]
    for bt in types:
        for name in tuple(vars(bt)):
            if name.startswith(("dun_sampler_", "_dun_sampler_")):
                delattr(bt, name)
    pbt = PackedBlockTypes.from_restype_list(
        old.chem_db, old.restype_set, types, pose.device
    )
    return attr.evolve(pose, packed_block_types=pbt)


def samplers(default_database, device):
    first = DunbrackParamResolver.from_database(default_database.scoring.dun, device)
    mapping = first.all_table_indices.copy()
    mapping.loc["ILE", "dun_table_name"] = mapping.loc["LEU", "dun_table_name"]
    second = attr.evolve(first, all_table_indices=mapping)
    return DunbrackChiSampler(first), DunbrackChiSampler(second)


def annotate(sampler, pose):
    for bt in pose.packed_block_types.active_block_types:
        sampler.annotate_residue_type(bt)
    sampler.annotate_packed_block_types(pose.packed_block_types)
    return pose.packed_block_types.dun_sampler_cache


def samples(sampler, pose):
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    return sampler.sample_chi_for_poses(pose, SetPackerTask.from_packer_task(task))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explicit_annotation", [False, True])
def test_shared_annotations_follow_resolver_identity(
    reverse, explicit_annotation, ubq_pdb, default_database, torch_device
):
    pose = fresh_pose(
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    )
    pair = samplers(default_database, torch_device)[:: -1 if reverse else 1]
    expected = []
    for sampler in pair:
        clean = fresh_pose(pose)
        cache = annotate(sampler, clean)
        expected.append((cache, samples(sampler, clean)))
    assert not torch.equal(
        expected[0][0].rottable_set_for_bt, expected[1][0].rottable_set_for_bt
    )
    assert any(not torch.equal(a, b) for a, b in zip(expected[0][1], expected[1][1]))
    for index in (0, 1, 0, 1):
        sampler = pair[index]
        if explicit_annotation:
            annotate(sampler, pose)
        actual_samples = samples(sampler, pose)
        actual = pose.packed_block_types.dun_sampler_cache
        for field in attr.fields(type(actual)):
            torch.testing.assert_close(
                getattr(actual, field.name),
                getattr(expected[index][0], field.name),
                rtol=0,
                atol=0,
            )
        for got, want in zip(actual_samples, expected[index][1]):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_library_metadata_stays_on_host(default_database, torch_device, monkeypatch):
    sampler = samplers(default_database, torch_device)[0]
    resolver = sampler.dun_param_resolver
    names = numpy.array([["ILE", "LEU", "DAL", "not-a-library"]], dtype=object)
    expected = (
        resolver._indices_from_names(resolver.all_table_indices, names, torch_device)
        .cpu()
        .tolist()[0]
    )

    def device_lookup(*args, **kwargs):
        raise AssertionError("Host library lookup must not create a device tensor")

    monkeypatch.setattr(DunbrackParamResolver, "_indices_from_names", device_lookup)
    assert [sampler._library_index(name) for name in names[0]] == expected


def test_sampler_equality_requires_a_sampler(default_database, torch_device):
    sampler = samplers(default_database, torch_device)[0]
    assert sampler == DunbrackChiSampler(sampler.dun_param_resolver)
    assert sampler != None  # noqa: E711
    assert sampler != hash(sampler)
