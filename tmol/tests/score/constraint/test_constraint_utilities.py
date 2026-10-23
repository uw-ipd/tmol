import pytest
import torch
from tmol import pose_stack_from_file, PoseStackBuilder
from tmol.score import (
    constrain_all_ca,
    constrain_closest_interchain_ca_pair,
    create_mainchain_coordinate_constraints,
)


def test_create_mainchain_coordinate_constraints(
    ubq_pdb, default_database, torch_device
):
    pose_stack1 = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )
    pose_stack10 = PoseStackBuilder.from_poses([pose_stack1] * 10, torch_device)

    pose_stack10 = create_mainchain_coordinate_constraints(pose_stack10)
    assert pose_stack10.constraint_set is not None

    torch.testing.assert_close(
        pose_stack10.constraint_set.constraint_params[:3, 1:4],
        pose_stack10.coords[0, :3, :],
    )


def test_constrain_all_ca(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )

    constrained = constrain_all_ca(pose_stack)

    assert constrained.constraint_set is not None
    assert constrained.constraint_set.constraint_params.shape[1] == 5
    assert torch.all(constrained.constraint_set.constraint_params[:, 4] == 0.5)


def _brute_force_closest_ca_pair(pose_stack, chain_a, chain_b):
    """Independent ground truth: scan every cross-chain CA-CA pair directly."""
    chain_labels = pose_stack.pdb_info.chain_labels[0]
    real = pose_stack.block_type_ind64[0].cpu().numpy() >= 0

    def ca_coords(chain):
        coords = []
        for block in range(pose_stack.max_n_blocks):
            if real[block] and chain_labels[block] == chain:
                block_type = pose_stack.block_type(0, block)
                if "CA" in block_type.atom_to_idx:
                    offset = int(pose_stack.block_coord_offset[0, block])
                    coords.append(
                        pose_stack.coords[0, offset + block_type.atom_to_idx["CA"]]
                    )
        return torch.stack(coords)

    distances = torch.cdist(ca_coords(chain_a), ca_coords(chain_b))
    return float(distances.min())


def test_constrain_closest_interchain_ca_pair_finds_true_minimum(
    pertuzumab_pdb, default_database, torch_device
):
    """The constrained distance matches an independent brute-force minimum."""
    pose_stack = pose_stack_from_file(
        pertuzumab_pdb,
        torch_device,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    expected_distance = _brute_force_closest_ca_pair(pose_stack, "C", "D")

    constrained = constrain_closest_interchain_ca_pair(pose_stack, "C", "D")

    assert constrained.constraint_set is not None
    assert constrained.constraint_set.constraint_atoms.shape[0] == 1
    actual_distance = float(constrained.constraint_set.constraint_params[0, 0])
    assert actual_distance == pytest.approx(expected_distance, abs=1e-4)

    atom_a, atom_b = constrained.constraint_set.constraint_atoms[0, :2].tolist()
    # both constrained atoms are CA atoms, one per chain
    type_a = pose_stack.block_type(atom_a[0], atom_a[1])
    type_b = pose_stack.block_type(atom_b[0], atom_b[1])
    assert atom_a[2] == type_a.atom_to_idx["CA"]
    assert atom_b[2] == type_b.atom_to_idx["CA"]


def test_constrain_closest_interchain_ca_pair_custom_std_dev(
    pertuzumab_pdb, default_database, torch_device
):
    """std_dev is stored as the second constraint parameter."""
    pose_stack = pose_stack_from_file(
        pertuzumab_pdb,
        torch_device,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    constrained = constrain_closest_interchain_ca_pair(
        pose_stack, "C", "D", std_dev=0.25
    )
    assert float(constrained.constraint_set.constraint_params[0, 1]) == pytest.approx(
        0.25
    )


def test_constrain_closest_interchain_ca_pair_merges_with_existing_constraints(
    pertuzumab_pdb, default_database, torch_device
):
    """Calling it again on an already-constrained pose accumulates constraints."""
    pose_stack = pose_stack_from_file(
        pertuzumab_pdb,
        torch_device,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    once = constrain_closest_interchain_ca_pair(pose_stack, "C", "D")
    twice = constrain_closest_interchain_ca_pair(once, "C", "D")

    assert once.constraint_set.constraint_atoms.shape[0] == 1
    assert twice.constraint_set.constraint_atoms.shape[0] == 2


def test_constrain_closest_interchain_ca_pair_raises_when_chain_missing(
    pertuzumab_pdb, default_database, torch_device
):
    """A chain letter absent from the pose raises ValueError."""
    pose_stack = pose_stack_from_file(
        pertuzumab_pdb,
        torch_device,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    with pytest.raises(ValueError, match="no residue with a CA atom"):
        constrain_closest_interchain_ca_pair(pose_stack, "C", "Z")
