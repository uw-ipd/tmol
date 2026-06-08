import numpy

# from tmol.pose.pdb_info import PDBInfo
from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_pdb_info_split(ubq_pdb, torch_device):
    n_poses = 4
    poses = [
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=20 + i)
        for i in range(n_poses)
    ]
    pose_stack = PoseStackBuilder.from_poses(poses, torch_device)
    pdb_info = pose_stack.pdb_info
    for i in range(n_poses):
        split_pdb_info = pdb_info.split(i)
        assert split_pdb_info.residue_labels.shape == (1, pose_stack.max_n_blocks)
        assert split_pdb_info.residue_insertion_codes.shape == (
            1,
            pose_stack.max_n_blocks,
        )
        assert split_pdb_info.chain_labels.shape == (1, pose_stack.max_n_blocks)
        assert split_pdb_info.atom_occupancy.shape == (1, pose_stack.max_n_pose_atoms)
        assert split_pdb_info.atom_b_factor.shape == (1, pose_stack.max_n_pose_atoms)

        i_pose = poses[i]
        i_pdb_info = i_pose.pdb_info
        numpy.testing.assert_equal(
            i_pdb_info.residue_labels,
            split_pdb_info.residue_labels[:, : i_pose.max_n_blocks],
        )
        numpy.testing.assert_equal(
            i_pdb_info.residue_insertion_codes,
            split_pdb_info.residue_insertion_codes[:, : i_pose.max_n_blocks],
        )
        numpy.testing.assert_equal(
            i_pdb_info.chain_labels,
            split_pdb_info.chain_labels[:, : i_pose.max_n_blocks],
        )
        numpy.testing.assert_equal(
            i_pdb_info.atom_occupancy,
            split_pdb_info.atom_occupancy[:, : i_pose.max_n_pose_atoms],
        )
        numpy.testing.assert_equal(
            i_pdb_info.atom_b_factor,
            split_pdb_info.atom_b_factor[:, : i_pose.max_n_pose_atoms],
        )
