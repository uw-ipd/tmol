import attrs
import numpy
import torch
import math
import pytest

from tmol.pose.constraint_set import ConstraintSet
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm

from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase

def test_constraint_set_empty_initialization(torch_device):
    cs = ConstraintSet.create_empty(torch_device, 1)

    assert cs.device == torch_device
    assert cs.n_poses == 1
    assert cs.constraint_function_inds.shape == (0,)
    assert cs.constraint_atoms.shape == (0, 4, 3)
    assert cs.constraint_params.shape == (0, 1)
    assert cs.constraint_num_unique_blocks.shape == (0,)
    assert cs.constraint_unique_blocks.shape == (0,3)
    assert len(cs.constraint_functions) == 0

def test_constraint_set_add_constraints(torch_device, ubq_pdb):
    n_poses = 2

    pose_stack = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses)
        ],
        torch_device,
    )

    cs = ConstraintSet.create_empty(torch_device, n_poses)
    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47

    cs = cs.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    assert cs.device == torch_device
    assert cs.n_poses == n_poses
    assert cs.constraint_function_inds.shape == (1,)
    assert cs.constraint_atoms.shape == (1, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[:, :2], cnstr_atoms)
    assert cs.constraint_params.shape == (1, 1)
    assert cs.constraint_num_unique_blocks.shape == (1,)
    assert cs.constraint_unique_blocks.shape == (1, 3)

def test_constraint_set_concatenate_constraints(torch_device, ubq_pdb):
    n_poses = 2

    pose_stack = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses)
        ],
        torch_device,
    )

    cs1 = ConstraintSet.create_empty(torch_device, n_poses)
    # a distance constraint
    cnstr_atoms1 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params1 = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms1[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms1[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    # print("cnstr_atoms1:", cnstr_atoms1)
    cnstr_params1[0, 0] = 1.47

    cs1 = cs1.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms1, cnstr_params1
    )

    cs2 = ConstraintSet.create_empty(torch_device, n_poses)
    # a distance constraint
    cnstr_atoms2 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params2 = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 6)
    res2_type = pose_stack.block_type(0, 7)
    cnstr_atoms2[0, 0] = torch.tensor([0, 6, res1_type.atom_to_idx["C"]])
    cnstr_atoms2[0, 1] = torch.tensor([0, 7, res2_type.atom_to_idx["N"]])
    # print("cnstr_atoms2:", cnstr_atoms2)
    cnstr_params2[0, 0] = 1.47

    cs2 = cs2.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms2, cnstr_params2
    )

    cs = ConstraintSet.concatenate([cs1, cs2], from_multiple_pose_stacks=False)
    # print("cs.constraint_atoms:", cs.constraint_atoms)

    assert cs.device == torch_device
    assert cs.n_poses == n_poses
    assert cs.constraint_function_inds.shape == (2,)
    assert cs.constraint_atoms.shape == (2, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[0:1, :2], cnstr_atoms1)
    torch.testing.assert_close(cs.constraint_atoms[1:2, :2], cnstr_atoms2)
    assert cs.constraint_params.shape == (2, 1)
    assert cs.constraint_num_unique_blocks.shape == (2,)
    assert cs.constraint_unique_blocks.shape == (2, 3)
    assert len(cs.constraint_functions) == 1

def test_constraint_set_concatenate_constraints_2(torch_device, ubq_pdb):
    n_poses_A = 2
    n_poses_B = 3

    pose_stack_A = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses_A)
        ],
        torch_device,
    )
    pose_stack_B = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses_B)
        ],
        torch_device,
    )

    cs1 = ConstraintSet.create_empty(torch_device, n_poses_A)
    # a distance constraint
    cnstr_atoms1 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params1 = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack_A.block_type(0, 3)
    res2_type = pose_stack_A.block_type(0, 4)
    cnstr_atoms1[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms1[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    # print("cnstr_atoms1:", cnstr_atoms1)
    cnstr_params1[0, 0] = 1.47

    cs1 = cs1.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms1, cnstr_params1
    )

    cs2 = ConstraintSet.create_empty(torch_device, n_poses_B)
    # a distance constraint
    cnstr_atoms2 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    shifted_cnstr_atoms2 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params2 = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack_B.block_type(2, 6)
    res2_type = pose_stack_B.block_type(2, 7)
    cnstr_atoms2[0, 0] = torch.tensor([2, 6, res1_type.atom_to_idx["C"]])
    cnstr_atoms2[0, 1] = torch.tensor([2, 7, res2_type.atom_to_idx["N"]])
    shifted_cnstr_atoms2[0, 0] = torch.tensor([4, 6, res1_type.atom_to_idx["C"]])
    shifted_cnstr_atoms2[0, 1] = torch.tensor([4, 7, res2_type.atom_to_idx["N"]])
    # print("cnstr_atoms2:", cnstr_atoms2)
    cnstr_params2[0, 0] = 1.47

    cs2 = cs2.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms2, cnstr_params2
    )

    cs = ConstraintSet.concatenate([cs1, cs2], from_multiple_pose_stacks=True)
    # print("cs.constraint_atoms:", cs.constraint_atoms)

    assert cs.device == torch_device
    assert cs.n_poses == n_poses_A + n_poses_B
    assert cs.constraint_function_inds.shape == (2,)
    assert cs.constraint_atoms.shape == (2, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[0:1, :2], cnstr_atoms1)
    torch.testing.assert_close(cs.constraint_atoms[1:2, :2], shifted_cnstr_atoms2)
    assert cs.constraint_params.shape == (2, 1)
    assert cs.constraint_num_unique_blocks.shape == (2,)
    assert cs.constraint_unique_blocks.shape == (2, 3)
    assert len(cs.constraint_functions) == 1