"""Tests for set_named_torsions and related kinematics utilities."""

from __future__ import annotations

import numpy
import pytest

from tmol.io import extended_pose_stack_from_sequences
from tmol.kinematics import EdgeType, FoldForest, set_named_torsions
from tmol.pose import get_named_torsions, get_torsion_names


def block_names(pose_stack, pose):
    pbt = pose_stack.packed_block_types
    return [
        pbt.active_block_types[int(bt_ind)].name
        for bt_ind in pose_stack.block_type_ind64[pose]
        if int(bt_ind) != -1
    ]


def real_coords(pose_stack, pose):
    n_blocks = int((pose_stack.block_type_ind64[pose] != -1).sum())
    last = n_blocks - 1
    bt_ind = int(pose_stack.block_type_ind64[pose, last])
    n_atoms = int(pose_stack.block_coord_offset64[pose, last]) + int(
        pose_stack.packed_block_types.n_atoms[bt_ind]
    )
    return pose_stack.coords[pose, :n_atoms].cpu().numpy()


def c_to_n_fold_forest(pose_stack):
    """Fold forest rooting each single-chain pose at its last residue."""
    n_poses = pose_stack.n_poses
    edges = numpy.full((n_poses, 2, 4), -1, dtype=int)
    for pose in range(n_poses):
        last = int((pose_stack.block_type_ind64[pose] != -1).sum()) - 1
        edges[pose, 0] = [EdgeType.root_jump, -1, last, -1]
        edges[pose, 1] = [EdgeType.polymer, last, 0, -1]
    return FoldForest.from_edges(edges)


def test_set_named_torsions_roundtrip(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    before = real_coords(pose_stack, 0)
    moved = set_named_torsions(pose_stack, 0, 1, "chi1", 62.5)

    assert get_named_torsions(moved, 0, 1, "chi1") == pytest.approx(62.5, abs=1e-3)
    assert pose_stack.coords is not moved.coords
    assert get_named_torsions(pose_stack, 0, 1, "chi1") != pytest.approx(62.5, abs=1e-3)

    # rooting at the first residue leaves everything before the bond in place
    n_before = int(moved.block_coord_offset64[0, 1])
    numpy.testing.assert_allclose(
        real_coords(moved, 0)[:n_before], before[:n_before], atol=1e-4
    )


def test_set_named_torsions_batch_roundtrip(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    blocks = [1, 2, 3]
    phis = [-57.0, -60.0, -63.0]
    psis = [-47.0, -45.0, -43.0]
    moved = set_named_torsions(pose_stack, [0] * 3, blocks, ["phi"] * 3, phis)
    moved = set_named_torsions(moved, [0] * 3, blocks, ["psi"] * 3, psis)

    for block, phi, psi in zip(blocks, phis, psis):
        measured = get_named_torsions(moved, 0, block)
        assert measured["phi"] == pytest.approx(phi, abs=1e-3)
        assert measured["psi"] == pytest.approx(psi, abs=1e-3)


def test_set_named_torsions_radians(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    target = numpy.radians(-71.0)
    moved = set_named_torsions(pose_stack, 0, 2, "chi1", target, degrees=False)

    assert get_named_torsions(moved, 0, 2, "chi1", degrees=False) == pytest.approx(
        target, abs=1e-5
    )


def test_set_named_torsions_absent_torsion_raises(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    # the nterm patch removes the down connection, and phi along with it
    assert "phi" not in get_torsion_names(pose_stack, 0, 0)
    with pytest.raises(ValueError, match="no torsion"):
        set_named_torsions(pose_stack, 0, 0, "phi", -60.0)


def test_set_named_torsions_undefined_torsion_raises(torch_device):
    pose_stack = extended_pose_stack_from_sequences(
        "AKLFG", device=torch_device, termini=False
    )

    # unpatched, residue 0 keeps phi, but it reaches a residue that is not there
    assert "phi" in get_torsion_names(pose_stack, 0, 0)
    assert numpy.isnan(get_named_torsions(pose_stack, 0, 0, "phi"))
    with pytest.raises(ValueError, match="undefined"):
        set_named_torsions(pose_stack, 0, 0, "phi", -60.0)


def test_named_torsions_agree_across_fold_forests(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    reversed_ff = c_to_n_fold_forest(pose_stack)

    targets = {"phi": -61.0, "psi": -43.0, "chi1": 58.0}
    names, values = list(targets), list(targets.values())
    default_moved = set_named_torsions(pose_stack, 0, 2, names, values)
    reversed_moved = set_named_torsions(
        pose_stack, 0, 2, names, values, fold_forest=reversed_ff
    )

    # both trees drive the torsion to the requested value
    for name, target in targets.items():
        assert get_named_torsions(default_moved, 0, 2, name) == pytest.approx(
            target, abs=1e-3
        )
        assert get_named_torsions(reversed_moved, 0, 2, name) == pytest.approx(
            target, abs=1e-3
        )

    # ... but they move opposite ends of the chain
    start = real_coords(pose_stack, 0)
    n_first = int(pose_stack.block_coord_offset64[0, 1])
    last_offset = int(pose_stack.block_coord_offset64[0, 4])
    numpy.testing.assert_allclose(
        real_coords(default_moved, 0)[:n_first], start[:n_first], atol=1e-4
    )
    numpy.testing.assert_allclose(
        real_coords(reversed_moved, 0)[last_offset:], start[last_offset:], atol=1e-4
    )
    assert not numpy.allclose(
        real_coords(default_moved, 0), real_coords(reversed_moved, 0), atol=1e-3
    )
