"""Batched covalent-group kinematics agree with independent scalar calls."""

import numpy
import pytest
import torch

from tmol.kinematics.compiled import forward_only_op, inverse_kin
from tmol.pack.rotamer._conjugated_chi_sampler import (
    ConjugatedChiSampler,
    _fold_group_conformers,
    _measure_member_dofs,
    _KINEMATICS_BATCH_ATOMS,
)
from tmol.pack.rotamer._single_residue_kinforest import (
    construct_single_residue_kinforest,
)
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _pose


def _single_tree(rot_kf, device):
    def tensor(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    return torch.stack(
        [
            tensor(numpy.r_[-1, rot_kf.id]),
            tensor(numpy.r_[0, rot_kf.doftype]),
            tensor(numpy.r_[0, rot_kf.parent + 1]),
            tensor(numpy.r_[0, rot_kf.frame_x + 1]),
            tensor(numpy.r_[0, rot_kf.frame_y + 1]),
            tensor(numpy.r_[0, rot_kf.frame_z + 1]),
        ],
        dim=1,
    )


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_group_and_member_batches_match_scalar_calls(fixture, dtype, torch_device):
    pose, _ = _pose(FIXTURES[fixture], torch_device)
    sampler = ConjugatedChiSampler()
    for group, columns, conformers in sampler.group_conformers(pose):
        _, one, offsets = sampler.group_coords(pose, group, columns, conformers[:1])
        rot_kf, _ = sampler._group_kinforest(pose, group)
        tree = _single_tree(rot_kf, torch_device)
        nodes = torch.tensor(rot_kf.nodes, dtype=torch.int32, device=torch_device)
        scans = torch.tensor(rot_kf.scans, dtype=torch.int32, device=torch_device)
        gens = torch.tensor(rot_kf.gens, dtype=torch.int32)
        original_order = torch.tensor(numpy.argsort(rot_kf.id), device=torch_device)
        for count in (0, 1, 7, _KINEMATICS_BATCH_ATOMS // int(offsets[-1]) + 1):
            dofs = one.to(dtype).repeat(count, 1, 1)
            # Give each independent copy distinct torsions and jump rotations.
            dofs[:, 1:, 3] += torch.arange(count, device=torch_device)[:, None] * 0.13
            folded = _fold_group_conformers(rot_kf, dofs)
            assert folded.shape == (count, int(offsets[-1]), 3)
            assert folded.dtype == dtype
            if not count:
                continue
            expected = torch.stack(
                [
                    forward_only_op(d, nodes, scans, gens, tree)[1:][original_order]
                    for d in dofs
                ]
            )
            torch.testing.assert_close(folded, expected, atol=2e-5, rtol=2e-5)
            for owner, block in enumerate(group.blocks):
                bt = pose.packed_block_types.active_block_types[
                    int(pose.block_type_ind[group.pose, block])
                ]
                construct_single_residue_kinforest(bt)
                mkf = bt.rotamer_kinforest
                member_tree = _single_tree(mkf, torch_device)
                member = folded[:, int(offsets[owner]) : int(offsets[owner + 1])]
                actual = _measure_member_dofs(mkf, member)
                expected_dofs = []
                order = torch.tensor(mkf.id, dtype=torch.int64, device=torch_device)
                for coordinates in member:
                    kco = torch.cat([coordinates.new_zeros((1, 3)), coordinates[order]])
                    expected_dofs.append(
                        inverse_kin(
                            kco,
                            member_tree[:, 2],
                            member_tree[:, 3],
                            member_tree[:, 4],
                            member_tree[:, 5],
                            member_tree[:, 1],
                        )[1:]
                    )
                torch.testing.assert_close(
                    actual, torch.stack(expected_dofs), atol=2e-5, rtol=2e-5
                )
                many = _KINEMATICS_BATCH_ATOMS // member.shape[1] + 1
                torch.testing.assert_close(
                    _measure_member_dofs(mkf, member[:1].expand(many, -1, -1)),
                    expected_dofs[0].unsqueeze(0).expand(many, -1, -1),
                    atol=2e-5,
                    rtol=2e-5,
                )
                assert _measure_member_dofs(mkf, member[:0]).shape == (
                    0,
                    member.shape[1],
                    9,
                )


def test_reused_sampler_keeps_real_pose_groups_independent(torch_device):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import FixedAAChiSampler, build_rotamers
    from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
    from tmol.pose import PoseStackBuilder
    from tmol.pose._conjugated_groups import (
        find_conjugated_groups,
        lockstep_group_for_block,
    )

    first, context = _pose(FIXTURES["biotin"], torch_device)
    second = first.clone()
    group = find_conjugated_groups(first)[0]
    bt = first.packed_block_types.active_block_types[
        int(first.block_type_ind[0, group.anchor])
    ]
    # Distinct anchor backbone angles: identical copies would hide a collision
    # keyed only by block number. This is a perturbed-geometry regression input.
    n = int(second.block_coord_offset[0, group.anchor]) + bt.atom_to_idx["N"]
    second.coords[0, n, 0] += 0.2
    library = create_dunbrack_sampler_from_database(
        context.parameter_database, torch_device
    )
    shared = ConjugatedChiSampler(library_sampler=library)

    def build(pose):
        task = PackerTask(pose, PackerPalette())
        task.add_conformer_sampler(library)
        task.add_conformer_sampler(FixedAAChiSampler())
        add_conjugated_group_sampler(task, pose, sampler=shared)
        task.restrict_to_repacking()
        return build_rotamers(
            pose,
            SetPackerTask.from_packer_task(task),
            context.parameter_database.chemical,
        )

    individual = [build(p)[1] for p in (first, second)]

    # The anchor row values must differ for this test to detect pose-key aliases.
    def anchor_coordinates(rotamers):
        first_rot = int(rotamers.rot_offset_for_block[0, group.anchor])
        lo = int(rotamers.coord_offset_for_rot[first_rot])
        return rotamers.coords[lo : lo + len(bt.atoms)]

    assert not torch.allclose(
        anchor_coordinates(individual[0]), anchor_coordinates(individual[1])
    )
    batch = PoseStackBuilder.from_poses([first, second], torch_device)
    batch, rotamers = build(batch)
    for pose, reference in enumerate(individual):
        torch.testing.assert_close(
            rotamers.n_rots_for_block[pose],
            reference.n_rots_for_block[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rotamers.coords[rotamers.pose_ind_for_atom == pose],
            reference.coords,
            rtol=2e-5,
            atol=2e-5,
        )
    ids = lockstep_group_for_block(batch, rotamers)
    for pose in range(2):
        assert bool((ids[pose, list(group.blocks)] == pose).all())
