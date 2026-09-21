"""Mirrored peptides must receive corresponding library conformers."""

from collections import defaultdict

import numpy
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, FixedAAChiSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.tests.score.test_mirror_image_scoring import _pose, MIRROR_PAIR


def build_mirror_rotamers(database, device, side, expanded=False):
    pose = _pose(f"{MIRROR_PAIR}_{side}", database, device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    for sampler in (
        IncludeCurrentSampler(),
        FixedAAChiSampler(),
        create_dunbrack_sampler_from_database(database, device),
    ):
        task.add_conformer_sampler(sampler)
    if expanded:
        task.or_expand_chi(0)
        task.or_expand_chi(1)
    return build_rotamers(pose, SetPackerTask.from_packer_task(task), database.chemical)


def conformer_groups(pose, rotamers):
    groups = defaultdict(list)
    coords = rotamers.coords.detach().cpu().numpy()
    elements = {a.name: a.element for a in pose.packed_block_types.chem_db.atom_types}
    metadata = []
    for rt in pose.packed_block_types.active_block_types:
        names = tuple(sorted(rt.atom_to_idx))
        heavy_names = [a.name for a in rt.atoms if elements[a.atom_type] != "H"]
        indices = numpy.array([rt.atom_to_idx[n] for n in sorted(heavy_names)])
        metadata.append((names, indices))
    for index, (block, ti, offset) in enumerate(
        zip(
            rotamers.block_ind_for_rot.cpu().tolist(),
            rotamers.block_type_ind_for_rot.cpu().tolist(),
            rotamers.coord_offset_for_rot.cpu().tolist(),
        )
    ):
        names, atoms = metadata[ti]
        groups[block, names].append((index, coords[offset + atoms]))
    return groups


@pytest.mark.parametrize("expanded", [False, True])
def test_mirror_library_conformer_counts_and_heavy_geometry(
    default_database, torch_device, expanded
):
    database = default_database.with_symmetric_gly()
    pairs = [
        build_mirror_rotamers(database, torch_device, side, expanded)
        for side in ("l", "d")
    ]
    left, right = [conformer_groups(*pair) for pair in pairs]
    assert left.keys() == right.keys()
    counts = {
        key[0]: (len(left[key]), len(right[key]))
        for key in left
        if len(left[key]) != len(right[key])
    }
    assert not counts, f"Reflected conformer counts differ by block: {counts}"
    for key in left:
        a = numpy.array([r[1] for r in left[key]])
        b = numpy.array([r[1] for r in right[key]])
        # A one-to-one assignment preserves multiplicity; nearest-neighbor
        # reuse would hide missing states. Hydrogens with equivalent chemistry
        # can exchange names under reflection, so compare named heavy atoms.
        cost = cdist(a.reshape(len(a), -1), -b.reshape(len(b), -1))
        rows, columns = linear_sum_assignment(cost)
        numpy.testing.assert_allclose(
            a[rows],
            -b[columns],
            atol=1e-4,
            rtol=0,
            err_msg=f"Block {key[0]} offered a nonmirrored conformer",
        )


@pytest.mark.parametrize("expanded", [False, True])
def test_shared_missing_backbone_resolves_each_library_orientation(
    default_database, torch_device, expanded
):
    import torch
    from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import (
        _table_indices,
    )

    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    left_table, right_table = _table_indices(
        sampler.dun_param_resolver, ("ARG", "DARG"), torch_device
    )

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    expansion = torch.zeros((2, 4), dtype=torch.int32, device=torch_device)
    if expanded:
        expansion[:, :2] = 1
    # Two chirality choices at the same physical residue share one pair of
    # unresolved backbone torsions. Defaults must be resolved per library,
    # without choosing either target's chirality during backbone measurement.
    counts, _, brt, chi = sampler.launch_rotamer_building(
        torch.zeros((1, 3), device=torch_device),
        integer([2]),
        integer([0]),
        torch.full((2, 4), -1, dtype=torch.int32, device=torch_device),
        integer([[0, left_table], [0, right_table]]),
        expansion,
        torch.zeros((2, 4, 1), device=torch_device),
        torch.zeros((2, 4), dtype=torch.int32, device=torch_device),
        torch.full((2,), 0.98, device=torch_device),
        integer([4, 4]),
    )
    assert int(counts[0]) == int(counts[1]) > 1
    left, right = chi[brt == 0].cpu().numpy(), -chi[brt == 1].cpu().numpy()
    circular_left = numpy.concatenate((numpy.cos(left), numpy.sin(left)), axis=1)
    circular_right = numpy.concatenate((numpy.cos(right), numpy.sin(right)), axis=1)
    rows, columns = linear_sum_assignment(cdist(circular_left, circular_right))
    numpy.testing.assert_allclose(
        circular_left[rows], circular_right[columns], atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("side", ["l", "d"])
def test_glycine_hydrogens_remain_on_opposite_sides_of_the_backbone(
    default_database, torch_device, side
):
    pose, rotamers = build_mirror_rotamers(default_database, torch_device, side)
    coords = rotamers.coords.detach().cpu().numpy()
    checked = 0
    for ti, offset in zip(
        rotamers.block_type_ind_for_rot.cpu().tolist(),
        rotamers.coord_offset_for_rot.cpu().tolist(),
    ):
        rt = pose.packed_block_types.active_block_types[ti]
        if rt.base_name != "GLY":
            continue
        xyz = coords[
            offset
            + numpy.array([rt.atom_to_idx[n] for n in ("CA", "N", "C", "HA2", "HA3")])
        ]
        ca, n, c, h2, h3 = xyz
        normal = numpy.cross(n - ca, c - ca)
        assert numpy.dot(normal, h2 - ca) * numpy.dot(normal, h3 - ca) < 0
        assert numpy.linalg.norm(h2 - h3) > 1.6
        numpy.testing.assert_allclose(
            numpy.linalg.norm(xyz[3:] - ca, axis=1), 1.09, rtol=0, atol=0.01
        )
        checked += 1
    assert checked >= 12
