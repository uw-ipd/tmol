import numpy
import torch

from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


def test_pose_stack_connection_ctor(ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res)
    p = PoseStack.one_structure_from_residues_and_connections(
        ubq_res, connections, torch_device
    )

    n_ubq_res = len(ubq_res)
    max_n_atoms = p.packed_block_types.max_n_atoms
    max_n_conn = max(
        len(rt.connections) for rt in p.packed_block_types.active_block_types
    )

    assert p.residue_coords.shape == (1, 1228, 3)
    assert p.coords.shape == p.residue_coords.shape
    assert p.block_coord_offset.shape == (1, n_ubq_res)
    assert p.inter_residue_connections.shape == (1, n_ubq_res, max_n_conn, 2)
    assert p.inter_block_bondsep.shape == (
        1,
        n_ubq_res,
        n_ubq_res,
        max_n_conn,
        max_n_conn,
    )
    assert p.block_type_ind.shape == (1, n_ubq_res)

    assert p.coords.device == torch_device
    assert p.block_coord_offset.device == torch_device
    assert p.inter_residue_connections.device == torch_device
    assert p.inter_block_bondsep.device == torch_device
    assert p.block_type_ind.device == torch_device
    assert p.device == torch_device

    ubq_res_block_types = p.packed_block_types.inds_for_res(ubq_res)
    nats_per_block = p.packed_block_types.n_atoms[ubq_res_block_types]
    nats_per_block_coord_offset = torch.cumsum(nats_per_block, 0)
    numpy.testing.assert_equal(
        nats_per_block_coord_offset[:-1].cpu().numpy(),
        p.block_coord_offset[0][1:].cpu().numpy(),
    )

    res_coords_gold = numpy.zeros(
        (nats_per_block_coord_offset[-1], 3), dtype=numpy.float32
    )
    for i, res in enumerate(ubq_res):
        i_offset = nats_per_block_coord_offset[i - 1] if i > 0 else 0
        res_coords_gold[(i_offset) : (i_offset + nats_per_block[i])] = res.coords

    numpy.testing.assert_allclose(
        res_coords_gold, p.residue_coords[0], atol=1e-5, rtol=1e-5
    )
    numpy.testing.assert_allclose(res_coords_gold, p.coords[0].cpu().numpy())


def test_pose_stack_one_structure_from_polymeric_residues_ctor(ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res)
    p_gold = PoseStack.one_structure_from_residues_and_connections(
        ubq_res, connections, torch_device
    )
    p_new = PoseStack.one_structure_from_polymeric_residues(ubq_res, torch_device)

    assert p_gold.residue_coords.shape == p_new.residue_coords.shape
    assert p_gold.coords.shape == p_new.coords.shape
    assert (
        p_gold.inter_residue_connections.shape == p_new.inter_residue_connections.shape
    )
    assert p_gold.inter_block_bondsep.shape == p_new.inter_block_bondsep.shape
    assert p_gold.block_type_ind.shape == p_new.block_type_ind.shape


def two_ubq_poses(ubq_res, torch_device):
    p1 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:40], torch_device)
    p2 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:60], torch_device)
    return PoseStack.from_poses([p1, p2], torch_device)


def test_pose_stack_create_inter_residue_connections(ubq_res, torch_device):
    connections_by_name = find_simple_polymeric_connections(ubq_res[:4])
    inter_residue_connections = PoseStack.create_inter_residue_connections(
        ubq_res[:4], connections_by_name, torch_device
    )

    assert inter_residue_connections.shape == (4, 2, 2)
    assert inter_residue_connections.device == torch_device

    assert inter_residue_connections[0, 0, 0] == -1
    assert inter_residue_connections[0, 0, 1] == -1

    assert inter_residue_connections[0, 1, 0] == 1
    assert inter_residue_connections[0, 1, 1] == 0
    assert inter_residue_connections[1, 0, 0] == 0
    assert inter_residue_connections[1, 0, 1] == 1

    assert inter_residue_connections[1, 1, 0] == 2
    assert inter_residue_connections[1, 1, 1] == 0
    assert inter_residue_connections[2, 0, 0] == 1
    assert inter_residue_connections[2, 0, 1] == 1

    assert inter_residue_connections[2, 1, 0] == 3
    assert inter_residue_connections[2, 1, 1] == 0
    assert inter_residue_connections[3, 0, 0] == 2
    assert inter_residue_connections[3, 0, 1] == 1

    assert inter_residue_connections[3, 1, 0] == -1
    assert inter_residue_connections[3, 1, 1] == -1


def test_pose_stack_resolve_bond_separation(ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res[1:4])
    bonds = PoseStack.determine_single_structure_inter_block_bondsep(
        ubq_res[1:4], connections, torch_device
    )
    assert bonds[0, 1, 1, 0] == 1
    assert bonds[1, 2, 1, 0] == 1
    assert bonds[1, 0, 0, 1] == 1
    assert bonds[2, 1, 0, 1] == 1
    assert bonds[0, 2, 1, 0] == 4
    assert bonds[2, 0, 0, 1] == 4


def test_concatenate_pose_stacks_ctor(ubq_res, torch_device):
    p1 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:40], torch_device)
    p2 = PoseStack.one_structure_from_polymeric_residues(ubq_res[:60], torch_device)
    poses = PoseStack.from_poses([p1, p2], torch_device)
    assert poses.block_type_ind.shape == (2, 60)
    max_n_atoms = poses.packed_block_types.max_n_atoms
    assert poses.coords.shape == (2, 959, 3)
    assert poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)


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
