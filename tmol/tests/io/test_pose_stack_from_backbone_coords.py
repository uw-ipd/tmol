"""Tests for pose_stack_from_backbone_coords."""
import numpy
import torch
import pytest

from tmol.io.pose_stack_from_backbone_coords import (
    pose_stack_from_backbone_coords,
    canonical_form_from_backbone_coords,
    canonical_ordering_for_backbone_coords,
    packed_block_types_for_backbone_coords,
    _paramdb_for_backbone_coords,
    _DEFAULT_AA_ORDER,
    _BACKBONE_ATOM_NAMES,
)


def _build_backbone_from_pdb(ubq_pdb, device):
    """Extract backbone (N/CA/C/O) coords and sequence from a PDB string.

    Returns (coords, res_types, chain_id) as torch tensors suitable for
    pose_stack_from_backbone_coords.  Uses the pdb_parsing / canonical_form
    path so the residue type mapping is authoritative.
    """
    from tmol.io.canonical_ordering import canonical_form_from_pdb
    from tmol.chemical.restypes import three2one

    co = canonical_ordering_for_backbone_coords()
    cf = canonical_form_from_pdb(co, ubq_pdb, device)

    n_poses, max_n_res, _, _ = cf.coords.shape

    # Build backbone (N, CA, C, O) coords from canonical form.
    # cf.res_types[p, r] is the index in co.restype_io_equiv_classes (-1 = pad).
    # For each residue type, look up the canonical slot for each bb atom.
    bb_coords = torch.full((n_poses, max_n_res, 4, 3), numpy.nan, dtype=torch.float32)
    for r_idx in range(max_n_res):
        rt_ind = int(cf.res_types[0, r_idx])
        if rt_ind < 0:
            continue
        name3 = co.restype_io_equiv_classes[rt_ind]
        at_map = co.restypes_atom_index_mapping[name3]
        for bb_slot, at_name in enumerate(_BACKBONE_ATOM_NAMES):
            if at_name in at_map:
                tmol_idx = at_map[at_name]
                bb_coords[0, r_idx, bb_slot] = cf.coords[0, r_idx, tmol_idx]

    # Map tmol restype indices → one-letter codes → aa_order indices.
    # Residues not in aa_order (e.g. HIS_D, CYD) are kept as HIS/CYS.
    aa_order = _DEFAULT_AA_ORDER
    res_types_out = torch.full((n_poses, max_n_res), -1, dtype=torch.int64)
    for r_idx in range(max_n_res):
        rt_ind = int(cf.res_types[0, r_idx])
        if rt_ind < 0:
            continue
        name3 = co.restype_io_equiv_classes[rt_ind]
        # For HIS_D / CYD fall back to canonical HIS / CYS
        name3_canonical = {"HIS_D": "HIS", "HIS_POS": "HIS", "CYD": "CYS"}.get(
            name3, name3
        )
        one = three2one(name3_canonical)
        if one and one in aa_order:
            res_types_out[0, r_idx] = aa_order.index(one)

    chain_id = cf.chain_id.to(torch.int32)
    return bb_coords.to(device), res_types_out.to(device), chain_id.to(device)


# ---------------------------------------------------------------------------
# Basic construction tests
# ---------------------------------------------------------------------------


def test_pose_stack_from_backbone_coords_shape(ubq_pdb, torch_device):
    """PoseStack constructed from backbone coords has the right n_poses and n_blocks."""
    coords, res_types, chain_id = _build_backbone_from_pdb(ubq_pdb, torch_device)
    n_poses, max_n_res, _, _ = coords.shape
    n_valid = int((res_types[0] >= 0).sum())

    ps = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)

    assert len(ps) == n_poses
    assert ps.max_n_blocks == n_valid


def test_pose_stack_from_backbone_coords_device(ubq_pdb, torch_device):
    coords, res_types, chain_id = _build_backbone_from_pdb(ubq_pdb, torch_device)
    ps = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)
    assert ps.coords.device == torch_device


def test_pose_stack_from_backbone_coords_single_pose_squeeze(ubq_pdb, torch_device):
    """Single-pose tensors (no batch dim) are accepted and unsqueezed internally."""
    coords, res_types, chain_id = _build_backbone_from_pdb(ubq_pdb, torch_device)
    # Remove batch dimension
    coords_1d = coords[0]         # (max_n_res, 4, 3)
    res_types_1d = res_types[0]   # (max_n_res,)
    chain_id_1d = chain_id[0]     # (max_n_res,)

    ps = pose_stack_from_backbone_coords(
        coords_1d, res_types_1d, chain_id_1d, torch_device
    )
    assert len(ps) == 1


def test_backbone_coords_side_chains_built(ubq_pdb, torch_device):
    """After construction, total atoms > 4*n_res (side chains were added)."""
    coords, res_types, chain_id = _build_backbone_from_pdb(ubq_pdb, torch_device)
    n_valid = int((res_types[0] >= 0).sum())

    ps = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)

    # Flat coords tensor has shape (n_poses, total_atoms, 3).
    # With side chains built, total_atoms >> 4 * n_valid.
    total_atoms = ps.coords.shape[1]
    assert total_atoms > 4 * n_valid, (
        f"Expected >4 atoms/residue after ideal-geometry build; "
        f"got {total_atoms} atoms for {n_valid} residues"
    )


# ---------------------------------------------------------------------------
# Multi-chain test
# ---------------------------------------------------------------------------


def test_pose_stack_from_backbone_coords_two_chains(ubq_pdb, torch_device):
    """Splitting a single chain into two produces the right n_blocks."""
    coords, res_types, chain_id = _build_backbone_from_pdb(ubq_pdb, torch_device)
    n_valid = int((res_types[0] >= 0).sum())
    split = n_valid // 2

    # Manually assign chain 1 to second half
    chain_id_split = chain_id.clone()
    chain_id_split[0, split:n_valid] = 1

    ps = pose_stack_from_backbone_coords(
        coords, res_types, chain_id_split, torch_device
    )
    assert len(ps) == 1
    assert ps.max_n_blocks == n_valid


# ---------------------------------------------------------------------------
# Memoization tests
# ---------------------------------------------------------------------------


def test_memoization_of_backbone_coords_paramdb():
    p1 = _paramdb_for_backbone_coords()
    p2 = _paramdb_for_backbone_coords()
    assert p1 is p2


def test_memoization_of_canonical_ordering():
    co1 = canonical_ordering_for_backbone_coords()
    co2 = canonical_ordering_for_backbone_coords()
    assert co1 is co2


def test_memoization_of_packed_block_types(torch_device):
    pbt1 = packed_block_types_for_backbone_coords(torch_device)
    pbt2 = packed_block_types_for_backbone_coords(torch_device)
    assert pbt1 is pbt2


def test_device_of_packed_block_types(torch_device):
    pbt = packed_block_types_for_backbone_coords(torch_device)
    assert pbt.device == torch_device


# ---------------------------------------------------------------------------
# Canonical form stability: padding with -1 is handled
# ---------------------------------------------------------------------------


def test_padding_residues_ignored(torch_device):
    """Padding positions (res_types == -1) produce no residues in the PoseStack."""
    n_poses, n_res = 1, 5
    aa_order = _DEFAULT_AA_ORDER
    ala_idx = aa_order.index("A")

    # Simple extended-strand backbone for 5 slots.
    # Non-degenerate so ideal CB building (cross products) doesn't produce NaN.
    # Approximate extended geometry: N-CA=1.46Å, CA-C=1.52Å, each residue +3.8Å along x.
    _bb = torch.tensor([
        [0.00, 0.00, 0.00],   # N
        [1.46, 0.00, 0.00],   # CA
        [1.92, 1.43, 0.00],   # C
        [1.53, 2.50, 0.00],   # O
    ], dtype=torch.float32)
    coords = torch.stack([
        _bb + torch.tensor([i * 3.8, 0.0, 0.0]) for i in range(n_res)
    ]).unsqueeze(0).to(torch_device)  # (1, 5, 4, 3)

    res_types = torch.full((n_poses, n_res), -1, dtype=torch.int64, device=torch_device)
    res_types[0, :3] = ala_idx   # first 3 are valid ALA; last 2 are padding
    chain_id = torch.zeros(n_poses, n_res, dtype=torch.int32, device=torch_device)

    ps = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)
    assert len(ps) == 1
    assert ps.max_n_blocks == 3
