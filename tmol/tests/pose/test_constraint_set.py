import attr
import torch

from tmol.pose.constraint_set import ConstraintSet
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb


def test_constraint_set_empty_initialization(torch_device):
    cs = ConstraintSet.create_empty(torch_device, 1)

    assert cs.device == torch_device
    assert cs.n_poses == 1
    assert cs.constraint_function_inds.shape == (0,)
    assert cs.constraint_atoms.shape == (0, 4, 3)
    assert cs.constraint_params.shape == (0, 1)
    assert cs.constraint_num_unique_blocks.shape == (0,)
    assert cs.constraint_unique_blocks.shape == (0, 3)
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
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    cs = cs.add_constraints(ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params)

    assert cs.device == torch_device
    assert cs.n_poses == n_poses
    assert cs.constraint_function_inds.shape == (1,)
    assert cs.constraint_atoms.shape == (1, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[:, :2], cnstr_atoms)
    assert cs.constraint_params.shape == (1, 2)
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
    cnstr_params1 = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms1[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms1[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params1[0, 0] = 1.47
    cnstr_params1[0, 1] = 0.1

    cs1 = cs1.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms1, cnstr_params1
    )

    cs2 = ConstraintSet.create_empty(torch_device, n_poses)
    # a distance constraint
    cnstr_atoms2 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params2 = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 6)
    res2_type = pose_stack.block_type(0, 7)
    cnstr_atoms2[0, 0] = torch.tensor([0, 6, res1_type.atom_to_idx["C"]])
    cnstr_atoms2[0, 1] = torch.tensor([0, 7, res2_type.atom_to_idx["N"]])
    cnstr_params2[0, 0] = 1.47
    cnstr_params2[0, 1] = 0.1

    cs2 = cs2.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms2, cnstr_params2
    )

    cs = ConstraintSet.concatenate([cs1, cs2], from_multiple_pose_stacks=False)

    assert cs.device == torch_device
    assert cs.n_poses == n_poses
    assert cs.constraint_function_inds.shape == (2,)
    assert cs.constraint_atoms.shape == (2, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[0:1, :2], cnstr_atoms1)
    torch.testing.assert_close(cs.constraint_atoms[1:2, :2], cnstr_atoms2)
    assert cs.constraint_params.shape == (2, 2)
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
    cnstr_params1 = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack_A.block_type(0, 3)
    res2_type = pose_stack_A.block_type(0, 4)
    cnstr_atoms1[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms1[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    # print("cnstr_atoms1:", cnstr_atoms1)
    cnstr_params1[0, 0] = 1.47
    cnstr_params1[0, 1] = 0.1

    cs1 = cs1.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms1, cnstr_params1
    )

    cs2 = ConstraintSet.create_empty(torch_device, n_poses_B)
    # a distance constraint
    cnstr_atoms2 = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    shifted_cnstr_atoms2 = torch.full(
        (1, 2, 3), 0, dtype=torch.int32, device=torch_device
    )
    cnstr_params2 = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack_B.block_type(2, 6)
    res2_type = pose_stack_B.block_type(2, 7)
    cnstr_atoms2[0, 0] = torch.tensor([2, 6, res1_type.atom_to_idx["C"]])
    cnstr_atoms2[0, 1] = torch.tensor([2, 7, res2_type.atom_to_idx["N"]])
    shifted_cnstr_atoms2[0, 0] = torch.tensor([4, 6, res1_type.atom_to_idx["C"]])
    shifted_cnstr_atoms2[0, 1] = torch.tensor([4, 7, res2_type.atom_to_idx["N"]])
    cnstr_params2[0, 0] = 1.47
    cnstr_params2[0, 1] = 0.1

    cs2 = cs2.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms2, cnstr_params2
    )

    cs = ConstraintSet.concatenate([cs1, cs2], from_multiple_pose_stacks=True)

    assert cs.device == torch_device
    assert cs.n_poses == n_poses_A + n_poses_B
    assert cs.constraint_function_inds.shape == (2,)
    assert cs.constraint_atoms.shape == (2, ConstraintSet.MAX_N_ATOMS, 3)
    torch.testing.assert_close(cs.constraint_atoms[0:1, :2], cnstr_atoms1)
    torch.testing.assert_close(cs.constraint_atoms[1:2, :2], shifted_cnstr_atoms2)
    assert cs.constraint_params.shape == (2, 2)
    assert cs.constraint_num_unique_blocks.shape == (2,)
    assert cs.constraint_unique_blocks.shape == (2, 3)
    assert len(cs.constraint_functions) == 1


def test_split_constraint_set(default_database, ubq_pdb, torch_device):
    n_poses = 4
    poses = [
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=20 + i)
        for i in range(n_poses)
    ]

    # let's create a bunch of harmonic constraints between adjacent C-N pairs
    # and set the distances to be slightly off from ideal, so that we generate
    # a non-zero constraint energy
    cst_energy = ConstraintEnergyTerm(default_database, torch_device)
    individual_constraint_energies = []
    for i in range(n_poses):
        cnstr_atoms1 = torch.full(
            (poses[i].max_n_blocks - 1, 2, 3), 0, dtype=torch.int32, device=torch_device
        )
        cnstr_params1 = torch.full(
            (poses[i].max_n_blocks - 1, 2), 0, dtype=torch.float32, device=torch_device
        )
        for j in range(poses[i].max_n_blocks - 1):

            ij_res1_type = poses[i].block_type(0, j)
            ij_res2_type = poses[i].block_type(0, j + 1)
            cnstr_atoms1[j, 0] = torch.tensor([0, j, ij_res1_type.atom_to_idx["C"]])
            cnstr_atoms1[j, 1] = torch.tensor([0, j + 1, ij_res2_type.atom_to_idx["N"]])
            cnstr_params1[j, 0] = 1.6  # instead of 1.47
            cnstr_params1[j, 1] = 0.1
        i_cst_set = ConstraintSet.create_empty(torch_device, 1).add_constraints(
            ConstraintEnergyTerm.harmonic, cnstr_atoms1, cnstr_params1
        )
        poses[i] = attr.evolve(poses[i], constraint_set=i_cst_set)
        i_whole_pose_energy = cst_energy.render_whole_pose_scoring_module(poses[i])
        i_whole_pose_energy_value = i_whole_pose_energy(poses[i].coords)
        individual_constraint_energies.append(i_whole_pose_energy_value)

    pose_stack = PoseStackBuilder.from_poses(poses, torch_device)
    cst_energy = ConstraintEnergyTerm(default_database, torch_device)
    whole_stack_energy = cst_energy.render_whole_pose_scoring_module(pose_stack)
    whole_stack_energy_value = whole_stack_energy(pose_stack.coords)
    torch.testing.assert_close(
        whole_stack_energy_value, torch.concat(individual_constraint_energies, dim=1)
    )

    # Now let's split the constraint set and PoseStack and rescore
    for i in range(n_poses):
        split_pose_stack = pose_stack.split(i)
        split_cst_energy = cst_energy.render_whole_pose_scoring_module(split_pose_stack)
        split_energy_value = split_cst_energy(split_pose_stack.coords)
        torch.testing.assert_close(
            split_energy_value, individual_constraint_energies[i]
        )
