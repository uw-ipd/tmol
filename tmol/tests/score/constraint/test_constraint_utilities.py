import torch
from tmol import (
    pose_stack_from_pdb,
)
from tmol.score.constraint.utility import (
    constrain_all_ca,
    create_mainchain_coordinate_constraints,
)
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_create_mainchain_coordinate_constraints(
    ubq_pdb, default_database, torch_device, capsys
):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack10 = PoseStackBuilder.from_poses([pose_stack1] * 10, torch_device)

    capsys.readouterr()
    pose_stack10 = create_mainchain_coordinate_constraints(pose_stack10)
    assert capsys.readouterr() == ("", "")
    assert pose_stack10.constraint_set is not None

    torch.testing.assert_close(
        pose_stack10.constraint_set.constraint_params[:3, 1:4],
        pose_stack10.coords[0, :3, :],
    )


def _ca_targets(pose_stack):
    targets = []
    for pose_ind in range(pose_stack.n_poses):
        for block_ind in range(pose_stack.max_n_blocks):
            if not pose_stack.is_real_block(pose_ind, block_ind):
                continue
            block_type = pose_stack.block_type(pose_ind, block_ind)
            if "CA" not in block_type.atom_to_idx:
                continue
            ca_ind = block_type.atom_to_idx["CA"]
            targets.append(
                (
                    pose_ind,
                    block_ind,
                    ca_ind,
                    pose_stack.coords[
                        pose_ind,
                        pose_stack.block_coord_offset64[pose_ind, block_ind] + ca_ind,
                    ],
                )
            )
    return targets


def test_constrain_all_ca_protein_is_non_mutating(ubq_pdb, torch_device):
    pose_stack = pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=0, residue_end=8
    )
    coords_before = pose_stack.coords.clone()
    targets = _ca_targets(pose_stack)

    constrained = constrain_all_ca(pose_stack)

    assert constrained is not pose_stack
    assert pose_stack.constraint_set is None
    torch.testing.assert_close(pose_stack.coords, coords_before)
    assert constrained.coords is pose_stack.coords
    assert constrained.constraint_set.constraint_atoms.shape[0] == len(targets)
    torch.testing.assert_close(
        constrained.constraint_set.constraint_params[:, 1:4],
        torch.stack([target[3] for target in targets]),
    )
    torch.testing.assert_close(
        constrained.constraint_set.constraint_params[:, 4],
        torch.full((len(targets),), 0.5, device=torch_device),
    )


def test_constrain_all_ca_skips_non_ca_blocks_and_retains_constraints(
    protein_dna_pdb, torch_device
):
    pose_stack = pose_stack_from_pdb(protein_dna_pdb, torch_device)
    pose_stack = create_mainchain_coordinate_constraints(pose_stack)
    original_constraint_set = pose_stack.constraint_set
    original_atoms = original_constraint_set.constraint_atoms.clone()
    original_params = original_constraint_set.constraint_params.clone()
    targets = _ca_targets(pose_stack)
    n_real_blocks = int(torch.count_nonzero(pose_stack.block_type_ind >= 0))

    assert 0 < len(targets) < n_real_blocks

    constrained = constrain_all_ca(pose_stack)

    assert pose_stack.constraint_set is original_constraint_set
    assert original_constraint_set.constraint_atoms.shape == original_atoms.shape
    torch.testing.assert_close(original_constraint_set.constraint_atoms, original_atoms)
    torch.testing.assert_close(
        original_constraint_set.constraint_params, original_params
    )

    n_existing = original_atoms.shape[0]
    assert constrained.constraint_set.constraint_atoms.shape[0] == (
        n_existing + len(targets)
    )
    torch.testing.assert_close(
        constrained.constraint_set.constraint_atoms[:n_existing], original_atoms
    )
    torch.testing.assert_close(
        constrained.constraint_set.constraint_params[:n_existing], original_params
    )
    torch.testing.assert_close(
        constrained.constraint_set.constraint_params[n_existing:, 1:4],
        torch.stack([target[3] for target in targets]),
    )
