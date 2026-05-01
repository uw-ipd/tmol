import os

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

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "atomworks", "ubq_atomworks.pt"
)


@pytest.fixture(scope="session")
def ubq_atomworks_data():
    """Load the ubiquitin atomworks fixture (device-agnostic, loaded once)."""
    return torch.load(_FIXTURE_PATH, weights_only=True)


@pytest.fixture
def ubq_atomworks(ubq_atomworks_data, torch_device):
    """Ubiquitin atomworks tensors on the test device."""
    return {k: v.to(torch_device) for k, v in ubq_atomworks_data.items()}


def test_pose_stack_from_atomworks_basic(ubq_atomworks, torch_device):
    """Test that we can build a PoseStack from ubiquitin atomworks tensors."""
    coords = ubq_atomworks["coords"]
    residue_type = ubq_atomworks["residue_type"]
    chain_iid = ubq_atomworks["chain_iid"]

    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    assert len(ps) == 1
    assert ps.max_n_blocks == 76
    assert ps.coords.device == torch_device


def test_canonical_form_from_atomworks(ubq_atomworks, torch_device):
    """Test that canonical form has correct shapes and non-NaN backbone coords."""
    coords = ubq_atomworks["coords"]
    residue_type = ubq_atomworks["residue_type"]
    chain_iid = ubq_atomworks["chain_iid"]

    cf = canonical_form_from_atomworks(coords, residue_type, chain_iid)
    assert cf.res_types.shape == (1, 76)
    assert cf.coords.shape[0] == 1
    assert cf.coords.shape[1] == 76

    co = canonical_ordering_for_atomworks()
    for r in range(76):
        rt_idx = residue_type[0, r].item()
        name3 = ATOMWORKS_NAME3S[rt_idx]
        for atname in ["N", "CA", "C"]:
            canon_idx = co.restypes_atom_index_mapping[name3][atname]
            assert not torch.isnan(
                cf.coords[0, r, canon_idx, 0]
            ).item(), f"Expected non-NaN coord for {name3} atom {atname} at res {r}"


def test_round_trip_atomworks_posestack(ubq_atomworks, torch_device):
    """Test that atomworks -> PoseStack -> atomworks is a faithful round trip."""
    coords = ubq_atomworks["coords"]
    residue_type = ubq_atomworks["residue_type"]
    chain_iid = ubq_atomworks["chain_iid"]

    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    rt_coords, rt_residue_type, rt_chain_iid = atomworks_from_pose_stack(ps)

    assert torch.equal(residue_type, rt_residue_type)
    assert torch.equal(chain_iid, rt_chain_iid)

    for r in range(coords.shape[1]):
        rt_idx = residue_type[0, r].item()
        name3 = ATOMWORKS_NAME3S[rt_idx]
        atoms = ATOMWORKS_ATOM37_NAMES[name3]
        for a, at_name in enumerate(atoms):
            if at_name == "" or at_name == "OXT":
                continue
            torch.testing.assert_close(
                coords[0, r, a],
                rt_coords[0, r, a],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Mismatch at res={r}, atom={at_name} (slot {a})",
            )


def test_multichain(ubq_atomworks, torch_device):
    """Test that multi-chain structures are handled correctly."""
    coords = ubq_atomworks["coords"]
    residue_type = ubq_atomworks["residue_type"]
    chain_iid = ubq_atomworks["chain_iid"].clone()

    # Split ubiquitin into two chains at residue 38
    chain_iid[0, 38:] = 1

    ps = pose_stack_from_atomworks(coords, residue_type, chain_iid)
    assert len(ps) == 1

    rt_coords, rt_residue_type, rt_chain_iid = atomworks_from_pose_stack(ps)
    assert torch.equal(chain_iid, rt_chain_iid)
    assert torch.equal(residue_type, rt_residue_type)


def test_protein_only_validation(torch_device):
    """Test that non-protein residue_type values raise ValueError."""
    coords = torch.zeros((1, 3, 37, 3), dtype=torch.float32, device=torch_device)
    chain_iid = torch.zeros((1, 3), dtype=torch.int64, device=torch_device)

    residue_type = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="protein only"):
        canonical_form_from_atomworks(coords, residue_type, chain_iid)

    residue_type = torch.tensor([[1, 22, 3]], dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="protein only"):
        canonical_form_from_atomworks(coords, residue_type, chain_iid)


def test_all_20_amino_acids(torch_device):
    """Test that all 20 standard amino acids can be round-tripped."""
    torch.manual_seed(42)

    n_res = 20
    residue_type = torch.arange(
        _ATOMWORKS_MIN_PROTEIN_IDX,
        _ATOMWORKS_MAX_PROTEIN_IDX + 1,
        dtype=torch.int64,
        device=torch_device,
    ).unsqueeze(0)

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
