"""Task masks constrain coupled geometry before rotamers are enumerated."""

import pytest
import torch

from tmol.io import pose_stack_from_biotite
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pose._conjugated_groups import find_conjugated_groups
from tmol.tests.pack.rotamer.test_group_constraints import (
    assert_geometry,
    crosslinked_lysines,
)
from tmol.tests.pack.test_conjugated_group_packing import _task, _pack_and_check_score


@pytest.fixture(params=["free", "cycle", "external"])
def group_pose(request, torch_device):
    return pose_stack_from_biotite(
        crosslinked_lysines(request.param),
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=503,
        return_context=True,
    )


def masked_task(pose, context, mode, fixed_owners):
    task, sampler = _task(pose, context.parameter_database, pose.device)
    group = find_conjugated_groups(pose)[0]
    mask = torch.zeros_like(pose.block_type_ind, dtype=torch.bool)
    mask[group.pose, [group.blocks[o] for o in fixed_owners]] = True
    if mode == "packing":
        task.disable_packing_by_block_mask(mask)
    else:
        task.disable_sampler_by_block_mask(sampler, mask)
    return task, sampler, group


@pytest.mark.parametrize("mode", ["packing", "sampler"])
@pytest.mark.parametrize("fixed_owners", [(0,), (1,), (2,), (0, 2), (0, 1, 2)])
def test_masks_preserve_frozen_members_and_group_geometry(
    group_pose, mode, fixed_owners
):
    pose, context = group_pose
    task, sampler, group = masked_task(pose, context, mode, fixed_owners)
    concrete = SetPackerTask.from_packer_task(task)
    pose, rotamers = build_rotamers(pose, concrete, context.parameter_database.chemical)
    counts = [int(rotamers.n_rots_for_block[0, b]) for b in group.blocks]
    for owner in fixed_owners:
        assert counts[owner] == 1
    active_counts = [n for o, n in enumerate(counts) if o not in fixed_owners]
    assert len(set(active_counts)) <= 1
    n_conf = max(counts)
    coordinates = pose.coords.expand(n_conf, -1, -1).clone()
    for owner, block in enumerate(group.blocks):
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        first = int(rotamers.rot_offset_for_block[0, block])
        starts = rotamers.coord_offset_for_rot[first : first + counts[owner]].long()
        indices = starts[:, None] + torch.arange(bt.n_atoms, device=pose.device)
        start = int(pose.block_coord_offset[0, block])
        actual = rotamers.coords[indices]
        coordinates[:, start : start + bt.n_atoms] = actual
        if owner in fixed_owners:
            torch.testing.assert_close(
                actual[0], pose.coords[0, start : start + bt.n_atoms], atol=1e-4, rtol=0
            )
    assert_geometry(pose, coordinates, group.blocks)
    if fixed_owners == (0, 1, 2):
        counts, mapping, samples = sampler.create_samples_for_poses(pose, concrete)
        assert not counts.any()
        assert mapping.numel() == 0
        assert samples == dict(groups=[], plan=[])


@pytest.mark.parametrize("fixed_owners", [(2,), (0, 2)])
def test_masked_group_budget_counts_active_and_background_rotamers(
    group_pose, fixed_owners
):
    pose, context = group_pose
    task, _, group = masked_task(pose, context, "packing", fixed_owners)
    # Two pendant three-state torsions + input = 10 conformers. The frozen
    # members cost one rotamer each, not ten copies of the same geometry.
    budget = (len(group) - len(fixed_owners)) * 10 + len(fixed_owners)
    task.set_chi_sample_budget(budget, budget)
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), context.parameter_database.chemical
    )
    counts = [int(rotamers.n_rots_for_block[0, b]) for b in group.blocks]
    assert counts == [1 if o in fixed_owners else 10 for o in range(len(group))]
    assert sum(counts) == budget


@pytest.mark.parametrize("fixed_owners", [(2,), (0, 2), (0, 1, 2)])
def test_masked_group_packing_preserves_geometry_and_energy(group_pose, fixed_owners):
    pose, context = group_pose
    task, _, group = masked_task(pose, context, "packing", fixed_owners)
    packed = _pack_and_check_score(
        pose, context.parameter_database, pose.device, task=task
    )
    assert_geometry(pose, packed.coords, group.blocks)


def test_reused_sampler_and_batched_masks_are_pose_local(group_pose):
    from tmol.pack import PackerPalette, PackerTask
    from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler
    from tmol.pose import PoseStackBuilder

    pose, context = group_pose
    _, sampler = _task(pose, context.parameter_database, pose.device)
    group = find_conjugated_groups(pose)[0]

    def build(stack, fixed_by_pose):
        task = PackerTask(stack, PackerPalette())
        task.add_conformer_sampler(sampler.library_sampler)
        add_conjugated_group_sampler(task, stack, sampler=sampler)
        task.restrict_to_repacking()
        mask = torch.zeros_like(stack.block_type_ind, dtype=torch.bool)
        for p, owners in enumerate(fixed_by_pose):
            mask[p, [group.blocks[o] for o in owners]] = True
        task.disable_packing_by_block_mask(mask)
        return build_rotamers(
            stack,
            SetPackerTask.from_packer_task(task),
            context.parameter_database.chemical,
        )[1]

    masks = [(2,), (0, 1, 2), (0, 2)]
    references = [build(pose, [owners]) for owners in masks]
    # Reusing the same PBT after all atoms were fixed must restore its earlier
    # movable axes, without relying on the order tasks were constructed.
    repeated = build(pose, [masks[0]])
    torch.testing.assert_close(
        repeated.coords, references[0].coords, atol=2e-5, rtol=2e-5
    )
    stack = PoseStackBuilder.from_poses([pose] * len(masks), pose.device)
    batched = build(stack, masks)
    for p, reference in enumerate(references):
        torch.testing.assert_close(
            batched.n_rots_for_block[p], reference.n_rots_for_block[0], atol=0, rtol=0
        )
        torch.testing.assert_close(
            batched.coords[batched.pose_ind_for_atom == p],
            reference.coords,
            atol=2e-5,
            rtol=2e-5,
        )


def test_member_without_its_own_chi_does_not_add_independent_fallback(group_pose):
    import attr
    from tmol.chemical import ResidueTypeSet
    from tmol.pose import PackedBlockTypes

    pose, context = group_pose
    group = find_conjugated_groups(pose)[0]
    source = pose.packed_block_types
    # Make a private sampling definition with a rigid linker. Its atoms still
    # move with upstream axes and must participate in every group conformer.
    linker = int(pose.block_type_ind[0, group.blocks[1]])
    types = [
        attr.evolve(rt, chi_samples=()) if i == linker else rt
        for i, rt in enumerate(source.active_block_types)
    ]
    restypes = ResidueTypeSet.from_restype_list(source.chem_db, types)
    pbt = PackedBlockTypes.from_restype_list(
        source.chem_db, restypes, types, pose.device
    )
    pose = attr.evolve(pose, packed_block_types=pbt)
    task, sampler = _task(pose, context.parameter_database, pose.device)
    assert sampler.defines_rotamers_for_rt(types[linker])
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), context.parameter_database.chemical
    )
    counts = [int(rotamers.n_rots_for_block[0, b]) for b in group.blocks]
    assert len(set(counts)) == 1


def test_library_free_group_preserves_entire_anchor(group_pose):
    from tmol.pack import PackerPalette, PackerTask
    from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
    from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler

    pose, context = group_pose
    group = find_conjugated_groups(pose)[0]
    task = PackerTask(pose, PackerPalette())
    sampler = ConjugatedChiSampler()
    add_conjugated_group_sampler(task, pose, sampler=sampler)
    task.restrict_to_repacking()
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), context.parameter_database.chemical
    )
    counts = [int(rotamers.n_rots_for_block[0, b]) for b in group.blocks]
    assert counts[0] == 1
    assert len(set(counts[1:])) == 1
    assert counts[1] > 1
    coordinates = pose.coords.expand(counts[1], -1, -1).clone()
    for owner, block in enumerate(group.blocks):
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        first = int(rotamers.rot_offset_for_block[0, block])
        starts = rotamers.coord_offset_for_rot[first : first + counts[owner]].long()
        indices = starts[:, None] + torch.arange(bt.n_atoms, device=pose.device)
        start = int(pose.block_coord_offset[0, block])
        coordinates[:, start : start + bt.n_atoms] = rotamers.coords[indices]
    anchor = group.anchor
    start = int(pose.block_coord_offset[0, anchor])
    bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, anchor])]
    actual = coordinates[:, start : start + bt.n_atoms]
    torch.testing.assert_close(
        actual,
        pose.coords[:, start : start + bt.n_atoms].expand_as(actual),
        atol=1e-4,
        rtol=0,
    )
    assert_geometry(pose, coordinates, group.blocks)
