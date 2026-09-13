"""Explicit polymer chi must reach the native sampler without a borrowed library."""

import itertools
import math

import attr
import numpy
import pytest
import torch

from tmol.database.chemical import ChiSamples
from tmol.io import pose_stack_from_pdb
from tmol.numeric import coord_dihedrals
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import FixedAAChiSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pose import PackedBlockTypes, PoseStackBuilder
from tmol.tests.pack.rotamer.dunbrack.test_sampler_parameter_identity import fresh_pose


def own_chi_pose(ubq_pdb, device, expanded=False, n_poses=1, gapped=False):
    pose = fresh_pose(
        pose_stack_from_pdb(ubq_pdb, device, residue_start=0, residue_end=12)
    )
    old = pose.packed_block_types
    samples = (ChiSamples("chi1", (-60.0, 60.0, 180.0), (), False),)
    if expanded:
        samples = (
            attr.evolve(samples[0], expansions=(20.0,)),
            ChiSamples("chi2", (-45.0, 45.0), (10.0,), False),
        )
    if gapped:
        samples = (ChiSamples("chi2", (-45.0, 45.0), (), False),)
    # Preserve valid ILE topology/coordinates, but explicitly supply its chi
    # without a library. Rebuild private types before constructing the task.
    types = [
        (
            attr.evolve(
                bt, base_name="XIL", dunbrack_reference=None, chi_samples=samples
            )
            if bt.base_name == "ILE"
            else bt
        )
        for bt in old.active_block_types
    ]
    pbt = PackedBlockTypes.from_restype_list(
        old.chem_db, old.restype_set, types, device
    )
    pose = attr.evolve(pose, packed_block_types=pbt)
    if n_poses > 1:
        pose = PoseStackBuilder.from_poses([pose] * n_poses, device)
    return pose, samples


def explicit_angles(samples):
    axes = []
    for sample in samples:
        offsets = [
            0.0,
            *itertools.chain.from_iterable((-x, x) for x in sample.expansions),
        ]
        axes.append(
            [
                math.radians(mean + offset)
                for mean in sample.samples
                for offset in offsets
            ]
        )
    return numpy.array(list(itertools.product(*axes)), dtype=numpy.float32)


def target_types(pose):
    return torch.tensor(
        [
            i
            for i, bt in enumerate(pose.packed_block_types.active_block_types)
            if bt.base_name == "XIL"
        ],
        device=pose.device,
    )


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_own_heavy_chi_rows_and_pose_masks(
    expanded, masked, ubq_pdb, default_database, torch_device
):
    pose, samples = own_chi_pose(ubq_pdb, torch_device, expanded, n_poses=2)
    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    mask = torch.ones_like(pose.block_type_ind, dtype=torch.bool)
    if masked:
        mask[1] = False
    task.add_conformer_sampler_by_block_mask(sampler, mask)
    concrete = SetPackerTask.from_packer_task(task)
    counts, ids, defining_atoms, chi = sampler.sample_chi_for_poses(pose, concrete)
    targets = (
        torch.isin(concrete.cons_bt_block_type, target_types(pose))
        & concrete.is_cons_bt_allowed
    )
    assert int(targets.sum()) == 2
    expected = explicit_angles(samples)
    for gbt in torch.nonzero(targets).flatten().tolist():
        active = not masked or int(concrete.cons_bt_pose[gbt]) == 0
        assert int(counts[gbt]) == (len(expected) if active else 0)
        rows = ids == gbt
        if not active:
            assert not bool(rows.any())
            continue
        numpy.testing.assert_allclose(
            chi[rows, : len(samples)].cpu(), expected, atol=1e-6
        )
        bt = pose.packed_block_types.active_block_types[
            int(concrete.cons_bt_block_type[gbt])
        ]
        expected_atoms = [bt.torsion_to_uaids[s.chi_dihedral][2][0] for s in samples]
        assert defining_atoms[rows, : len(samples)].tolist() == [expected_atoms] * len(
            expected
        )


def test_own_chi_count_obeys_explicit_budget(ubq_pdb, default_database, torch_device):
    pose, samples = own_chi_pose(ubq_pdb, torch_device, expanded=True)
    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    mask = torch.isin(pose.block_type_ind, target_types(pose))
    task.add_conformer_sampler_by_block_mask(sampler, mask)
    task.set_chi_sample_budget(len(explicit_angles(samples)) - 1, 1)
    with pytest.raises(RuntimeError, match="Sampling budget"):
        sampler.sample_chi_for_poses(pose, SetPackerTask.from_packer_task(task))


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("empty_libraries", [False, True])
def test_own_chi_builds_requested_coordinates(
    gapped, empty_libraries, ubq_pdb, default_database, torch_device
):
    if empty_libraries:
        default_database = attr.evolve(
            default_database,
            scoring=attr.evolve(
                default_database.scoring,
                dun=attr.evolve(
                    default_database.scoring.dun,
                    dun_lookup=(),
                    rotameric_libraries=(),
                    semi_rotameric_libraries=(),
                ),
            ),
        )
    pose, samples = own_chi_pose(ubq_pdb, torch_device, gapped=gapped)
    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    task.add_conformer_sampler(FixedAAChiSampler())
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    targets = torch.isin(pose.block_type_ind, target_types(pose))
    assert int(targets.sum()) == 1
    p, block = torch.nonzero(targets)[0].tolist()
    bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[p, block])]
    count = int(rotamers.n_rots_for_block[p, block])
    assert count == len(explicit_angles(samples))
    first = int(rotamers.rot_offset_for_block[p, block])
    starts = rotamers.coord_offset_for_rot[first : first + count].long()
    atom = torch.arange(bt.n_atoms, device=torch_device)
    coords = rotamers.coords[starts[:, None] + atom]
    assert bool(torch.isfinite(coords).all())
    chi_atoms = torch.tensor(
        [a[0] for a in bt.torsion_to_uaids[samples[0].chi_dihedral]],
        device=torch_device,
    )
    measured = (
        coord_dihedrals(*coords[:, chi_atoms].double().unbind(dim=1)).cpu().numpy()
    )
    expected = explicit_angles(samples)[:, 0]
    numpy.testing.assert_allclose(numpy.sin(measured - expected), 0.0, atol=2e-5)
    numpy.testing.assert_allclose(numpy.cos(measured - expected), 1.0, atol=2e-5)
    if gapped:
        frozen_atoms = torch.tensor(
            [a[0] for a in bt.torsion_to_uaids["chi1"]], device=torch_device
        )
        original_start = int(pose.block_coord_offset[p, block])
        original = pose.coords[p, original_start + frozen_atoms].double()
        original_angle = coord_dihedrals(*original.unsqueeze(0).unbind(dim=1))
        frozen = coord_dihedrals(*coords[:, frozen_atoms].double().unbind(dim=1))
        torch.testing.assert_close(
            frozen, original_angle.expand_as(frozen), atol=2e-5, rtol=0
        )
