"""Unit tests for get_torsion_names and get_named_torsions in tmol/pose/_util.py."""

from __future__ import annotations

import numpy
import pytest

from tmol.io import EXTENDED_BACKBONE_TORSIONS, extended_pose_stack_from_sequences
from tmol.pose import get_named_torsions, get_torsion_names

# ── get_torsion_names ─────────────────────────────────────────────────────────


def test_get_torsion_names_interior_residue_has_backbone_and_chi(torch_device):
    """An interior protein residue reports phi, psi, omega and its chi torsions."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    # LYS is block 1: interior, has 4 chi torsions
    names = get_torsion_names(pose_stack, 0, 1)
    assert "phi" in names
    assert "psi" in names
    assert "omega" in names
    assert "chi1" in names
    assert "chi2" in names
    assert "chi3" in names
    assert "chi4" in names


def test_get_torsion_names_nterm_patch_removes_phi(torch_device):
    """The N-terminal patch removes phi from the torsion list."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    names = get_torsion_names(pose_stack, 0, 0)  # ALA:nterm
    assert "phi" not in names
    assert "psi" in names
    assert "omega" in names


def test_get_torsion_names_cterm_patch_removes_psi_and_omega(torch_device):
    """The C-terminal patch removes psi and omega from the torsion list."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    names = get_torsion_names(pose_stack, 0, 4)  # GLY:cterm
    assert "phi" in names
    assert "psi" not in names
    assert "omega" not in names


def test_get_torsion_names_padding_block_raises(torch_device):
    """Requesting torsion names for a padding slot (bt_ind == -1) raises ValueError."""
    pose_stack = extended_pose_stack_from_sequences(["AA", "A"], device=torch_device)
    # Pose 1 has 1 real block; block index 1 is padding (bt_ind == -1)
    with pytest.raises(ValueError, match="not a real block"):
        get_torsion_names(pose_stack, 1, 1)


def test_get_torsion_names_returns_list_in_database_order(torch_device):
    """Torsion names are returned in the same order as they appear in the block type."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    pbt = pose_stack.packed_block_types
    bt_ind = int(pose_stack.block_type_ind64[0, 1])
    bt = pbt.active_block_types[bt_ind]
    expected = [tor.name for tor in bt.torsions]
    assert get_torsion_names(pose_stack, 0, 1) == expected


# ── get_named_torsions – return-type dispatch ─────────────────────────────────


def test_get_named_torsions_scalar_name_returns_float(torch_device):
    """scalar pose, scalar block, scalar name → bare float."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    val = get_named_torsions(pose_stack, 0, 1, "chi1")
    assert isinstance(val, float)


def test_get_named_torsions_no_name_returns_dict(torch_device):
    """scalar pose, scalar block, name=None → dict keyed by every torsion name."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    result = get_named_torsions(pose_stack, 0, 1)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(get_torsion_names(pose_stack, 0, 1))


def test_get_named_torsions_list_of_names_returns_dict(torch_device):
    """scalar pose, scalar block, list-of-names → dict with only those keys."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    result = get_named_torsions(pose_stack, 0, 1, ["phi", "psi"])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"phi", "psi"}


def test_get_named_torsions_non_scalar_poses_returns_list_of_lists(torch_device):
    """Non-scalar (list) poses argument → list[pose][block] of dicts."""
    pose_stack = extended_pose_stack_from_sequences(
        ["AKLFG", "AKLFG"], device=torch_device
    )
    result = get_named_torsions(pose_stack, [0, 1], [1, 1])
    assert isinstance(result, list)
    assert len(result) == pose_stack.n_poses


def test_get_named_torsions_no_args_returns_all_poses_and_blocks(torch_device):
    """Omitting poses and blocks returns a list[pose][block] covering everything."""
    pose_stack = extended_pose_stack_from_sequences(
        ["AKLFG", "AKLFG"], device=torch_device
    )
    result = get_named_torsions(pose_stack)
    assert len(result) == pose_stack.n_poses
    for pose_row in result:
        assert len(pose_row) == pose_stack.max_n_blocks
        assert isinstance(pose_row[0], dict)


# ── get_named_torsions – numerical correctness ────────────────────────────────


def test_get_named_torsions_ideal_backbone_correct(torch_device):
    """Backbone torsions of an extended-conformation pose are at their target values."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    targets = EXTENDED_BACKBONE_TORSIONS["alpha"]
    # LYS (block 1) is interior; phi, psi, and omega are all resolvable
    measured = get_named_torsions(pose_stack, 0, 1)
    for name, target in targets.items():
        if name in measured and not numpy.isnan(measured[name]):
            delta = (measured[name] - target + 180.0) % 360.0 - 180.0
            assert abs(delta) < 0.01, (
                f"backbone torsion {name!r}: measured {measured[name]:.3f}, "
                f"expected {target:.3f}"
            )


def test_get_named_torsions_degrees_default(torch_device):
    """By default torsion values are returned in degrees."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    psi_deg = get_named_torsions(pose_stack, 0, 1, "psi")
    # Extended-conformation psi is close to 135° (protein target), well outside ±π
    assert abs(psi_deg) > numpy.pi, "expected degrees, got a value in radian range"


def test_get_named_torsions_radians_consistent_with_degrees(torch_device):
    """degrees=False and degrees=True report the same dihedral in different units."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    deg = get_named_torsions(pose_stack, 0, 1, "psi", degrees=True)
    rad = get_named_torsions(pose_stack, 0, 1, "psi", degrees=False)
    assert deg == pytest.approx(numpy.degrees(rad), abs=1e-5)


def test_get_named_torsions_nan_for_absent_neighbor(torch_device):
    """phi on residue 0 is nan when no N-terminal neighbor exists (termini=False)."""
    pose_stack = extended_pose_stack_from_sequences(
        "AKLFG", device=torch_device, termini=False
    )
    # Without the N-terminal patch, phi is still defined on block 0...
    assert "phi" in get_torsion_names(pose_stack, 0, 0)
    # ...but cannot be measured because the preceding residue is absent
    val = get_named_torsions(pose_stack, 0, 0, "phi")
    assert numpy.isnan(val)


def test_get_named_torsions_dict_nan_for_absent_torsions(torch_device):
    """In dict mode, torsions crossing absent neighbors appear as nan, not absent."""
    pose_stack = extended_pose_stack_from_sequences(
        "AKLFG", device=torch_device, termini=False
    )
    result = get_named_torsions(pose_stack, 0, 0)
    assert "phi" in result
    assert numpy.isnan(result["phi"])


# ── get_named_torsions – padding and error handling ───────────────────────────


def test_get_named_torsions_padding_block_silently_skipped_when_implicit(torch_device):
    """Padding blocks are skipped (not an error) when blocks is not explicitly given."""
    pose_stack = extended_pose_stack_from_sequences(["AA", "A"], device=torch_device)
    # Pose 1 has 1 real block; asking for all blocks should skip the padding slot
    result = get_named_torsions(pose_stack, 1)
    # The padding block's slot produces an empty dict, not an error
    assert isinstance(result, list)
    # The real block has torsion entries; the padding slot dict is empty
    assert len(result[0]) == pose_stack.max_n_blocks


def test_get_named_torsions_explicit_padding_block_raises(torch_device):
    """Explicitly requesting a padding block raises ValueError."""
    pose_stack = extended_pose_stack_from_sequences(["AA", "A"], device=torch_device)
    with pytest.raises(ValueError, match="not a real block"):
        get_named_torsions(pose_stack, 1, 1, "phi")


def test_get_named_torsions_unknown_torsion_name_raises(torch_device):
    """Requesting a torsion name absent from the block type raises ValueError."""
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    # ALA:nterm has no phi (N-terminal patch removed it)
    with pytest.raises(ValueError, match="no torsion"):
        get_named_torsions(pose_stack, 0, 0, "phi")


def test_get_named_torsions_mismatched_pose_block_lengths_raises(torch_device):
    """List poses and blocks of different lengths raise ValueError."""
    pose_stack = extended_pose_stack_from_sequences(
        ["AKLFG", "AKLFG"], device=torch_device
    )
    with pytest.raises(ValueError, match="same length"):
        get_named_torsions(pose_stack, [0, 1], [1, 2, 3], "phi")


# ── get_named_torsions – paired vs Cartesian-product selection ────────────────


def test_get_named_torsions_paired_lists_are_not_cartesian_product(torch_device):
    """Same-length poses and blocks lists are zipped (paired), not crossed."""
    pose_stack = extended_pose_stack_from_sequences(
        ["AKLFG", "AKLFG"], device=torch_device
    )
    # Paired: (pose=0, block=1) and (pose=1, block=2)
    result = get_named_torsions(pose_stack, [0, 1], [1, 2])
    # A Cartesian product would have 4 entries in the requests; a paired selection
    # has 2.  The list-of-lists result should be populated only at (0,1) and (1,2).
    assert isinstance(result, list)
    assert len(result) == pose_stack.n_poses
    # (0,1) and (1,2) are populated; cross-product positions (0,2) and (1,1) are not
    assert result[0][1]  # (pose 0, block 1) has torsion entries
    assert result[1][2]  # (pose 1, block 2) has torsion entries
    assert not result[0][2]  # (pose 0, block 2) is absent from the paired selection
    assert not result[1][1]  # (pose 1, block 1) is absent from the paired selection
