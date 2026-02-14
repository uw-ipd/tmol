import torch
import pytest

from tmol.io.pose_stack_from_atomworks import (
    pose_stack_from_atomworks,
    canonical_form_from_atomworks,
    atomworks_from_pose_stack,
    canonical_ordering_for_atomworks,
    packed_block_types_for_atomworks,
    _paramdb_for_atomworks,
    ATOMWORKS_NAME3S,
    ATOMWORKS_ATOM37_NAMES,
    _ATOMWORKS_MIN_PROTEIN_IDX,
    _ATOMWORKS_MAX_PROTEIN_IDX,
)


def _make_synthetic_atomworks_batch(n_poses, n_res, device):
    """Create a synthetic atomworks batch with random coords for ALA/GLY/VAL.

    Returns (coords, residue_type, chain_iid) tensors.
    OXT is never populated: tmol handles terminal oxygen via its patch system,
    so it is excluded from the forward mapping and cannot round-trip.
    """
    # Pick a few AA types: ALA=1, GLY=8, VAL=20
    aa_choices = torch.tensor([1, 8, 20], dtype=torch.int64, device=device)
    residue_type = aa_choices[
        torch.randint(0, len(aa_choices), (n_poses, n_res), device=device)
    ]

    # Build coords: random non-zero coords for real atoms, zeros for absent
    coords = torch.zeros((n_poses, n_res, 37, 3), dtype=torch.float32, device=device)
    for p in range(n_poses):
        for r in range(n_res):
            rt_idx = residue_type[p, r].item()
            name3 = ATOMWORKS_NAME3S[rt_idx]
            atoms = ATOMWORKS_ATOM37_NAMES[name3]
            for a, at_name in enumerate(atoms):
                if at_name == "" or at_name == "OXT":
                    continue
                coords[p, r, a] = torch.randn(3, device=device) * 10.0

    # Single chain per pose
    chain_iid = torch.zeros((n_poses, n_res), dtype=torch.int64, device=device)

    return coords, residue_type, chain_iid


def test_pose_stack_from_atomworks_basic(torch_device):
    """Test that we can build a PoseStack from synthetic atomworks tensors."""
    coords, residue_type, chain_iid = _make_synthetic_atomworks_batch(
        n_poses=2, n_res=5, device=torch_device
    )
    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    assert len(ps) == 2
    assert ps.max_n_blocks == 5
    assert ps.coords.device == torch_device


def test_canonical_form_from_atomworks(torch_device):
    """Test that canonical form has correct shapes and non-NaN backbone coords."""
    coords, residue_type, chain_iid = _make_synthetic_atomworks_batch(
        n_poses=1, n_res=3, device=torch_device
    )
    cf = canonical_form_from_atomworks(coords, residue_type, chain_iid)
    assert cf.res_types.shape == (1, 3)
    assert cf.coords.shape[0] == 1
    assert cf.coords.shape[1] == 3

    # All protein residues should have valid N, CA, C coords (backbone)
    # which should not be NaN since we populated them
    co = canonical_ordering_for_atomworks()
    for r in range(3):
        rt_idx = residue_type[0, r].item()
        name3 = ATOMWORKS_NAME3S[rt_idx]
        for atname in ["N", "CA", "C"]:
            canon_idx = co.restypes_atom_index_mapping[name3][atname]
            assert not torch.isnan(
                cf.coords[0, r, canon_idx, 0]
            ).item(), f"Expected non-NaN coord for {name3} atom {atname}"


def test_round_trip_atomworks_posestack(torch_device):
    """Test that atomworks -> PoseStack -> atomworks is a faithful round trip."""
    coords, residue_type, chain_iid = _make_synthetic_atomworks_batch(
        n_poses=2, n_res=6, device=torch_device
    )

    # Forward: atomworks -> PoseStack
    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)

    # Reverse: PoseStack -> atomworks
    rt_coords, rt_residue_type, rt_chain_iid = atomworks_from_pose_stack(ps)

    # Residue types should match
    assert torch.equal(residue_type, rt_residue_type)

    # Chain IDs should match
    assert torch.equal(chain_iid, rt_chain_iid)

    # Coords should match for all real non-OXT atoms (within tolerance).
    # OXT is excluded from the forward mapping (tmol handles it via patches).
    for p in range(coords.shape[0]):
        for r in range(coords.shape[1]):
            rt_idx = residue_type[p, r].item()
            name3 = ATOMWORKS_NAME3S[rt_idx]
            atoms = ATOMWORKS_ATOM37_NAMES[name3]
            for a, at_name in enumerate(atoms):
                if at_name == "" or at_name == "OXT":
                    continue
                torch.testing.assert_close(
                    coords[p, r, a],
                    rt_coords[p, r, a],
                    atol=1e-4,
                    rtol=1e-4,
                    msg=f"Mismatch at pose={p}, res={r}, atom={at_name} (slot {a})",
                )


def test_multichain(torch_device):
    """Test that multi-chain structures are handled correctly."""
    n_poses, n_res = 1, 8
    coords, residue_type, chain_iid = _make_synthetic_atomworks_batch(
        n_poses=n_poses, n_res=n_res, device=torch_device
    )
    # Two chains: 0,0,0,0,1,1,1,1
    chain_iid[0, :4] = 0
    chain_iid[0, 4:] = 1

    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    assert len(ps) == 1

    # Round-trip
    rt_coords, rt_residue_type, rt_chain_iid = atomworks_from_pose_stack(ps)
    assert torch.equal(chain_iid, rt_chain_iid)
    assert torch.equal(residue_type, rt_residue_type)


def test_protein_only_validation(torch_device):
    """Test that non-protein residue_type values raise ValueError."""
    coords = torch.zeros((1, 3, 37, 3), dtype=torch.float32, device=torch_device)
    chain_iid = torch.zeros((1, 3), dtype=torch.int64, device=torch_device)

    # Index 0 (mask) is out of protein range
    residue_type = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="protein only"):
        canonical_form_from_atomworks(coords, residue_type, chain_iid)

    # Index 22 (RNA) is out of range
    residue_type = torch.tensor([[1, 22, 3]], dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="protein only"):
        canonical_form_from_atomworks(coords, residue_type, chain_iid)


def test_all_20_amino_acids(torch_device):
    """Test that all 20 standard amino acids can be round-tripped."""
    n_res = 20
    residue_type = torch.arange(
        _ATOMWORKS_MIN_PROTEIN_IDX,
        _ATOMWORKS_MAX_PROTEIN_IDX + 1,
        dtype=torch.int64,
        device=torch_device,
    ).unsqueeze(
        0
    )  # [1, 20]

    coords = torch.zeros((1, n_res, 37, 3), dtype=torch.float32, device=torch_device)
    for r in range(n_res):
        rt_idx = residue_type[0, r].item()
        name3 = ATOMWORKS_NAME3S[rt_idx]
        atoms = ATOMWORKS_ATOM37_NAMES[name3]
        for a, at_name in enumerate(atoms):
            if at_name == "" or at_name == "OXT":
                continue
            coords[0, r, a] = torch.randn(3, device=torch_device) * 10.0

    chain_iid = torch.zeros((1, n_res), dtype=torch.int64, device=torch_device)

    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    rt_coords, rt_residue_type, rt_chain_iid = atomworks_from_pose_stack(ps)

    assert torch.equal(residue_type, rt_residue_type)

    for r in range(n_res):
        rt_idx = residue_type[0, r].item()
        name3 = ATOMWORKS_NAME3S[rt_idx]
        atoms = ATOMWORKS_ATOM37_NAMES[name3]
        for a, at_name in enumerate(atoms):
            if at_name == "" or at_name == "OXT":
                continue
            torch.testing.assert_close(
                coords[0, r, a], rt_coords[0, r, a], atol=1e-4, rtol=1e-4
            )


def test_memoization_of_paramdb():
    """Test that _paramdb_for_atomworks is memoized."""
    p1 = _paramdb_for_atomworks()
    p2 = _paramdb_for_atomworks()
    assert p1 is p2


def test_memoization_of_canonical_ordering():
    """Test that canonical_ordering_for_atomworks is memoized."""
    co1 = canonical_ordering_for_atomworks()
    co2 = canonical_ordering_for_atomworks()
    assert co1 is co2


def test_memoization_of_packed_block_types(torch_device):
    """Test that packed_block_types_for_atomworks is memoized."""
    pbt1 = packed_block_types_for_atomworks(torch_device)
    pbt2 = packed_block_types_for_atomworks(torch_device)
    assert pbt1 is pbt2


def test_device_of_packed_block_types(torch_device):
    """Test that PBT is on the correct device."""
    pbt = packed_block_types_for_atomworks(torch_device)
    assert pbt.device == torch_device
