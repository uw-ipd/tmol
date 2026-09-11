"""Regressions for pose-local and residue-local group sampling identities."""

from types import SimpleNamespace

import numpy as np
import torch
import pytest

from tmol.database.chemical import ChiSamples
from tmol.pack.rotamer._chi_budget import _budgeted_chi_samples, apply_chi_sample_budget
from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
from tmol.pack.rotamer._conjugated_groups import (
    GroupSamplingTopology,
    group_sampled_chi,
)
from tmol.pose._conjugated_groups import ConjugatedGroup


def sample(name, values=(0.0, 120.0, 240.0), expansions=()):
    return ChiSamples(
        chi_dihedral=name, samples=values, expansions=expansions, is_proton=False
    )


def test_budget_preserves_owner_when_chi_names_repeat():
    samples = [sample("chi1"), sample("chi1", values=(30.0, 60.0, 90.0))]
    kept = _budgeted_chi_samples(samples, [10, 2], 3, 3)
    assert kept == ((1, samples[1]),)


def test_independent_library_does_not_remove_child_chi():
    samples = [sample("chi1"), sample("chi2")]
    kept = _budgeted_chi_samples(samples, [1, 2], 81, 81, library_size=9)
    assert kept == tuple(enumerate(samples))
    # Single-block callers still omit chi that their own library supplies.
    assert apply_chi_sample_budget(samples, [1, 2], 81, 81, n_library_chi=1) == (
        samples[1],
    )


def test_budget_retains_a_proton_placement_and_rejects_oversize_library():
    import attr

    proton = attr.evolve(sample("chi2", expansions=(20,)), is_proton=True)
    kept = _budgeted_chi_samples([proton], [2], 1, 1)
    assert len(kept) == 1
    assert kept[0][1].samples == (0.0,)
    assert kept[0][1].expansions == ()
    with pytest.raises(ValueError, match="required 11 library states"):
        _budgeted_chi_samples([], [], 10, 10, library_size=11)


def test_group_budget_keeps_correct_residue_and_linkage():
    anchor = SimpleNamespace(
        n_atoms=1, torsions=[SimpleNamespace(name="chi1"), SimpleNamespace(name="chi2")]
    )
    children = [
        SimpleNamespace(
            n_atoms=2,
            chi_samples=[sample("chi1", values=values)],
            torsion_to_uaids={"chi1": [(0, -1, -1)] * 2 + [(1, -1, -1)] * 2},
        )
        for values in ((0.0, 120.0, 240.0), (30.0, 60.0, 90.0))
    ]
    pose = SimpleNamespace(
        packed_block_types=SimpleNamespace(active_block_types=[anchor, *children]),
        block_type_ind=torch.tensor([[0, 1, 2]]),
    )
    # Child 1 is further from the root and must freeze before child 2.
    rkd = SimpleNamespace(
        bfto_2_orig=np.array([0, 3, 4, 1, 2]), preds=np.array([-1, 4, 1, 0, 3])
    )
    topology = GroupSamplingTopology(
        [anchor, *children],
        np.array([0, 1, 3, 5]),
        {},
        rkd,
        {frozenset((1, 2)): 2, frozenset((3, 4)): 4},
    )
    group = ConjugatedGroup(0, (0, 1, 2), ((0, 0, 1, 0), (1, 1, 2, 0)))
    kept = group_sampled_chi(group, pose, 27, 27, library_size=3, topology=topology)
    assert kept == [(2, children[1].chi_samples[0])]


def test_anchor_library_keeps_identical_block_numbers_in_different_poses(
    torch_device, monkeypatch
):
    import tmol.pack.rotamer._conjugated_chi_sampler as conjugated

    monkeypatch.setattr(
        conjugated,
        "group_sampling_topology",
        lambda *args: SimpleNamespace(
            block_types=[
                SimpleNamespace(
                    torsion_to_uaids={
                        "chi1": [(0, -1, -1), (1, -1, -1), (2, -1, -1), (3, -1, -1)]
                    }
                )
            ],
            movable_axes={frozenset((1, 2)): 2},
        ),
    )

    class Library:
        def sample_chi_for_poses(self, pose, library_task):
            # An observer must never see the caller's task temporarily changed.
            torch.testing.assert_close(allowed, original_allowed)
            selected = library_task.per_block_conformer_sampler_allowed[:, :, 0]
            torch.testing.assert_close(
                selected,
                torch.tensor([[True, False], [True, False]], device=torch_device),
            )
            return (
                None,
                torch.tensor([0, 1], device=torch_device),
                torch.tensor([[2], [3]], device=torch_device),
                torch.tensor([[0.25], [1.25]], device=torch_device),
            )

    library = Library()
    sampler = ConjugatedChiSampler(library_sampler=library)
    allowed = torch.tensor([[[False], [True]], [[False], [True]]], device=torch_device)
    original_allowed = allowed.clone()
    task = SimpleNamespace(
        conformer_sampler_index={id(library): 0},
        per_block_conformer_sampler_allowed=allowed,
        cons_bt_pose=torch.tensor([0, 1], device=torch_device),
        cons_bt_block=torch.tensor([0, 0], device=torch_device),
        cons_bt_block_type=torch.tensor([0, 1], device=torch_device),
    )
    pose = SimpleNamespace(
        block_type_ind=torch.tensor([[0, 2], [1, 2]], device=torch_device)
    )
    groups = [ConjugatedGroup(p, (0, 1), ((0, 0, 1, 0),)) for p in range(2)]
    result = sampler.anchor_library_chi(pose, task, groups)
    assert set(result) == {(0, 0), (1, 0)}
    np.testing.assert_array_equal(result[(0, 0)][1], [[0.25]])
    np.testing.assert_array_equal(result[(1, 0)][1], [[1.25]])
    torch.testing.assert_close(allowed, original_allowed)


def test_kinforest_cache_is_scoped_to_packed_block_types(monkeypatch):
    import tmol.pack.rotamer._single_residue_kinforest as kin
    import tmol.pack.rotamer._conjugated_chi_sampler as conjugated

    calls = []

    def construct(types, links, anchor, kinforest_data=None):
        result = (object(), object())
        calls.append(result)
        return result

    monkeypatch.setattr(kin, "construct_block_group_kinforest", construct)
    monkeypatch.setattr(
        conjugated,
        "group_sampling_topology",
        lambda group, pose: SimpleNamespace(
            block_types=pose.packed_block_types.active_block_types,
            kinforest=object(),
            offsets=object(),
        ),
    )
    sampler = ConjugatedChiSampler()
    group = ConjugatedGroup(0, (0, 1), ((0, 0, 1, 0),))
    poses = [
        SimpleNamespace(
            packed_block_types=SimpleNamespace(active_block_types=[object(), object()]),
            block_type_ind=torch.tensor([[0, 1]]),
        )
        for _ in range(2)
    ]
    first = sampler._group_kinforest(poses[0], group)
    assert sampler._group_kinforest(poses[0], group) is first
    assert sampler._group_kinforest(poses[1], group) is not first
    assert len(calls) == 2
    for conn in range(33):
        other = ConjugatedGroup(0, (0, 1), ((0, conn, 1, 0),))
        sampler._group_kinforest(poses[0], other)
    assert len(poses[0].packed_block_types.conjugated_kinforest_cache) == 32
    assert sampler._group_kinforest(poses[0], group) is not first


def test_lockstep_rejects_mismatched_counts_before_native_scoring(
    monkeypatch, torch_device
):
    import tmol.pose._conjugated_groups as groups
    import pytest

    group = ConjugatedGroup(0, (0, 1), ((0, 0, 1, 0),))
    monkeypatch.setattr(groups, "find_conjugated_groups", lambda pose: [group])
    pose = SimpleNamespace(n_poses=1, max_n_blocks=2)
    rotamers = SimpleNamespace(
        n_rots_for_block=torch.tensor([[3, 2]], device=torch_device)
    )
    with pytest.raises(ValueError, match="differing rotamer counts"):
        groups.lockstep_group_for_block(pose, rotamers)
    rotamers.n_rots_for_block[:] = 3
    result = groups.lockstep_group_for_block(pose, rotamers)
    torch.testing.assert_close(
        result, torch.zeros((1, 2), dtype=torch.int32, device=torch_device)
    )


def test_group_collapse_accepts_exclusive_end_offset(torch_device):
    from tmol.pack.rotamer._conjugated_groups import collapse_group_rotamers

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    rots = SimpleNamespace(
        coords=torch.zeros((4, 3), device=torch_device),
        block_ind_for_rot=tensor([0, 0, 1, 1]),
        n_rots_for_block=tensor([[2, 2, 0]]),
        rot_offset_for_block=tensor([[0, 2, 4]]),
        rot_offset_for_pose=tensor([0]),
        n_rots_for_pose=tensor([4]),
        pose_for_rot=tensor([0, 0, 0, 0]),
        block_type_ind_for_rot=tensor([0, 0, 1, 1]),
    )
    group = ConjugatedGroup(0, (0, 1), ((0, 0, 1, 0),))
    collapse, _, _ = collapse_group_rotamers(None, rots, [group])
    torch.testing.assert_close(
        collapse.compact_rot_offset_for_block, tensor([[0, 2, 3]])
    )
    torch.testing.assert_close(collapse.compact_n_rots_for_block, tensor([[2, 1, 0]]))
    torch.testing.assert_close(collapse.compact_n_rots_for_pose, tensor([3]))


def test_constrained_library_projection_preserves_first_distinct_states(monkeypatch):
    import tmol.pack.rotamer._conjugated_chi_sampler as conjugated

    anchor = SimpleNamespace(
        chi_samples=(),
        torsion_to_uaids={
            "chi1": [(0, -1, -1), (1, -1, -1), (2, -1, -1), (3, -1, -1)],
            "chi2": [(1, -1, -1), (2, -1, -1), (3, -1, -1), (4, -1, -1)],
        },
    )
    child = SimpleNamespace(chi_samples=())
    pose = SimpleNamespace(
        packed_block_types=SimpleNamespace(active_block_types=[anchor, child]),
        block_type_ind=torch.tensor([[0, 1]]),
    )
    group = ConjugatedGroup(0, (0, 1), ((0, 0, 1, 0),))
    topology = GroupSamplingTopology(
        [anchor, child],
        np.array([0, 5, 6]),
        {},
        SimpleNamespace(bfto_2_orig=np.arange(6), preds=np.arange(6) - 1),
        {frozenset((1, 2)): 2},
    )
    monkeypatch.setattr(conjugated, "find_conjugated_groups", lambda pose: [group])
    monkeypatch.setattr(conjugated, "group_sampling_topology", lambda *args: topology)
    library = {
        (0, 0): (
            np.tile([2, 3], (4, 1)),
            np.array([[2.0, 1.0], [1.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
        )
    }
    sampler = ConjugatedChiSampler(library_sampler=object(), include_current=False)
    _, columns, conformers = sampler.group_conformers(pose, library, (4, 4))[0]
    assert columns == [(0, "chi1", 1, 2)]
    np.testing.assert_array_equal(conformers, [[2.0], [1.0]])
    with pytest.raises(ValueError, match="required 2 library states"):
        sampler.group_conformers(pose, library, (2, 2))


def test_topology_cache_separates_external_constraints_and_is_bounded(monkeypatch):
    import attr
    import tmol.pack.rotamer._conjugated_groups as conjugated

    pose = SimpleNamespace(
        packed_block_types=SimpleNamespace(), block_type_ind=torch.tensor([[0, 1]])
    )
    monkeypatch.setattr(conjugated, "_group_sampling_topology", lambda *args: object())
    group = ConjugatedGroup(0, (0, 1), ((0, 0, 1, 0),))
    free = conjugated.group_sampling_topology(group, pose)
    attached = attr.evolve(group, external_links=((1, 1, 2, 0),))
    fixed = conjugated.group_sampling_topology(attached, pose)
    assert fixed is not free
    # Which external pose block supplies the same constraint does not alter
    # the group's topology; an external owner's/port's identity does.
    equivalent = attr.evolve(group, external_links=((1, 1, 3, 7),))
    assert conjugated.group_sampling_topology(equivalent, pose) is fixed
    for port in range(33):
        attached = attr.evolve(group, external_links=((1, port, 2, 0),))
        conjugated.group_sampling_topology(attached, pose)
    assert len(pose.packed_block_types.conjugated_sampling_topology_cache) == 32
    assert conjugated.group_sampling_topology(group, pose) is not free
