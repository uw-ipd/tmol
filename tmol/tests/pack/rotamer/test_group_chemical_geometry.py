"""Every offered group conformer must retain chemical geometry and set its chi."""

import itertools
import numpy
import pytest
import torch

from tmol.numeric import coord_dihedrals
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pose._conjugated_groups import find_conjugated_groups
from tmol.pose._util import _resolve_uaid
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _pose, _task


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_group_conformers_preserve_all_bonds_angles_and_chirality(
    fixture, torch_device
):
    pose, context = _pose(FIXTURES[fixture], torch_device)
    task, sampler = _task(pose, context.parameter_database, torch_device)
    concrete = SetPackerTask.from_packer_task(task)
    pose, rotamers = build_rotamers(pose, concrete, context.parameter_database.chemical)
    groups = find_conjugated_groups(pose)
    anchor_chi = sampler.anchor_library_chi(pose, concrete, groups)
    enumerated = sampler.group_conformers(pose, anchor_chi)
    for group, columns, targets in enumerated:
        source, sampled, bonds, types = [], [], set(), []
        offsets = [0]
        pose_to_group = {}
        for block in group.blocks:
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[group.pose, block])
            ]
            types.append(bt)
            first = int(rotamers.rot_offset_for_block[group.pose, block])
            count = int(rotamers.n_rots_for_block[group.pose, block])
            assert count in (1, len(targets))
            starts = rotamers.coord_offset_for_rot[first : first + count].long()
            atom = torch.arange(bt.n_atoms, device=torch_device)
            sampled.append(
                rotamers.coords[starts[:, None] + atom].expand(len(targets), -1, -1)
            )
            start = int(pose.block_coord_offset[group.pose, block])
            source.append(pose.coords[group.pose, start : start + bt.n_atoms])
            pose_to_group.update(
                {
                    group.pose * pose.max_n_pose_atoms + start + a: offsets[-1] + a
                    for a in range(bt.n_atoms)
                }
            )
            bonds.update(
                tuple(sorted((offsets[-1] + int(a), offsets[-1] + int(b))))
                for a, b in bt.bond_indices
            )
            offsets.append(offsets[-1] + bt.n_atoms)
        for a, ac, b, bc in group.links:
            bonds.add(
                tuple(
                    sorted(
                        (
                            offsets[a] + int(types[a].ordered_connection_atoms[ac]),
                            offsets[b] + int(types[b].ordered_connection_atoms[bc]),
                        )
                    )
                )
            )
        source = torch.cat(source).double()
        sampled = torch.cat(sampled, dim=1).double()
        pairs = torch.tensor(sorted(bonds), dtype=torch.int64, device=torch_device)
        expected_lengths = (source[pairs[:, 0]] - source[pairs[:, 1]]).norm(dim=-1)
        lengths = (sampled[:, pairs[:, 0]] - sampled[:, pairs[:, 1]]).norm(dim=-1)
        torch.testing.assert_close(
            lengths, expected_lengths.expand_as(lengths), atol=1e-4, rtol=0
        )

        neighbors = [set() for _ in range(len(source))]
        for a, b in bonds:
            neighbors[a].add(b)
            neighbors[b].add(a)
        triples = [
            (a, center, b)
            for center, near in enumerate(neighbors)
            for a, b in itertools.combinations(sorted(near), 2)
        ]
        angle_atoms = torch.tensor(triples, dtype=torch.int64, device=torch_device)

        def cosine(coords):
            a, c, b = (coords[..., angle_atoms[:, j], :] for j in range(3))
            ac, bc = a - c, b - c
            return (ac * bc).sum(dim=-1) / (ac.norm(dim=-1) * bc.norm(dim=-1))

        actual_angles = cosine(sampled)
        torch.testing.assert_close(
            actual_angles, cosine(source).expand_as(actual_angles), atol=1e-4, rtol=0
        )

        tetrahedra = [sorted(near) for near in neighbors if len(near) == 4]
        if tetrahedra:
            tetra = torch.tensor(tetrahedra, dtype=torch.int64, device=torch_device)

            def volumes(coords):
                a, b, c, d = (coords[..., tetra[:, j], :] for j in range(4))
                return ((a - d) * torch.linalg.cross(b - d, c - d, dim=-1)).sum(dim=-1)

            expected = volumes(source)
            real_centers = expected.abs() > 1e-3
            actual = volumes(sampled)[:, real_centers]
            torch.testing.assert_close(
                actual.sign(),
                expected[real_centers].sign().expand_as(actual),
                atol=0,
                rtol=0,
            )

        for col, (owner, name, _b, _c) in enumerate(columns):
            # Resolve through the pose's own UAID path, independently of the
            # sampler's group-local resolver and kinforest numbering.
            atoms = [
                pose_to_group[_resolve_uaid(pose, group.pose, group.blocks[owner], u)]
                for u in types[owner].torsion_to_uaids[name]
            ]
            assert all(
                tuple(sorted(edge)) in bonds for edge in zip(atoms[:-1], atoms[1:])
            ), name
            actual = coord_dihedrals(*(sampled[:, a] for a in atoms))
            expected = torch.tensor(
                targets[:, col], dtype=torch.float64, device=torch_device
            )
            difference = (actual - expected + numpy.pi) % (2 * numpy.pi) - numpy.pi
            assert float(difference.abs().max()) < 2e-4, (fixture, owner, name)
