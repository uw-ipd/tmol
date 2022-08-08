import numpy
import torch

from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_n_poses(ubq_40_60_pose_stack):
    assert ubq_40_60_pose_stack.n_poses == 2


def test_max_n_blocks(ubq_40_60_pose_stack):
    assert ubq_40_60_pose_stack.max_n_blocks == 60


def test_max_n_atoms(ubq_40_60_pose_stack):
    assert (
        ubq_40_60_pose_stack.max_n_atoms
        == ubq_40_60_pose_stack.packed_block_types.max_n_atoms
    )


def test_max_n_block_atoms(ubq_40_60_pose_stack):
    assert (
        ubq_40_60_pose_stack.max_n_block_atoms
        == ubq_40_60_pose_stack.packed_block_types.max_n_atoms
    )


def test_max_n_pose_atoms(ubq_res, ubq_40_60_pose_stack):
    actual_n_atoms = sum(res.coords.shape[0] for res in ubq_res[:60])
    assert ubq_40_60_pose_stack.max_n_pose_atoms == actual_n_atoms


def test_n_ats_per_pose_block(ubq_40_60_pose_stack):
    n_ats_per_block_gold = torch.zeros((2, 60), dtype=torch.int32)
    for i in range(2):
        for j, res in enumerate(ubq_40_60_pose_stack.residues[i]):
            n_ats_per_block_gold[i, j] = res.coords.shape[0]
    numpy.testing.assert_equal(
        n_ats_per_block_gold, ubq_40_60_pose_stack.n_ats_per_block.cpu().numpy()
    )


def test_real_atoms(ubq_40_60_pose_stack):
    max_n_pose_atoms = ubq_40_60_pose_stack.max_n_pose_atoms
    real_ats_gold = torch.zeros((2, max_n_pose_atoms), dtype=bool)
    n_ats_per_pose = torch.sum(ubq_40_60_pose_stack.n_ats_per_block, dim=1).cpu()
    real_ats_gold[0, : n_ats_per_pose[0]] = 1
    real_ats_gold[1, : n_ats_per_pose[1]] = 1

    numpy.testing.assert_equal(
        real_ats_gold.numpy(), ubq_40_60_pose_stack.real_atoms.cpu().numpy()
    )


def test_expand_coords(ubq_40_60_pose_stack, torch_device):
    poses = ubq_40_60_pose_stack
    expanded_coords_gold = torch.zeros(
        (2, poses.max_n_blocks, poses.max_n_block_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    real_expanded_coords_gold = torch.zeros(
        (2, poses.max_n_blocks, poses.max_n_block_atoms),
        dtype=torch.bool,
        device=torch_device,
    )
    n_ats_per_block = poses.n_ats_per_block
    for i in range(2):
        for j in range(len(poses.residues[i])):
            ij_nats = n_ats_per_block[i, j]
            ij_offset = poses.block_coord_offset[i, j]
            expanded_coords_gold[i, j, :ij_nats] = poses.coords[
                i, ij_offset : (ij_offset + ij_nats)
            ]
            real_expanded_coords_gold[i, j, :ij_nats] = True

    expanded_coords, real_expanded_coords = poses.expand_coords()
    numpy.testing.assert_equal(
        expanded_coords_gold.cpu().numpy(), expanded_coords.cpu().numpy()
    )
    numpy.testing.assert_equal(
        real_expanded_coords.cpu().numpy(), real_expanded_coords.cpu().numpy()
    )
