import numpy
import torch

from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_pose_stack_builder_connection_ctor(ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res)
    p = PoseStackBuilder.one_structure_from_residues_and_connections(
        ubq_res, connections, torch_device
    )

    n_ubq_res = len(ubq_res)
    # max_n_atoms = p.packed_block_types.max_n_atoms
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


def test_pose_stack_builder_one_structure_from_polymeric_residues_ctor(
    ubq_res, torch_device
):
    connections = find_simple_polymeric_connections(ubq_res)
    p_gold = PoseStackBuilder.one_structure_from_residues_and_connections(
        ubq_res, connections, torch_device
    )
    p_new = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res, torch_device
    )

    assert p_gold.residue_coords.shape == p_new.residue_coords.shape
    assert p_gold.coords.shape == p_new.coords.shape
    assert (
        p_gold.inter_residue_connections.shape == p_new.inter_residue_connections.shape
    )
    assert p_gold.inter_block_bondsep.shape == p_new.inter_block_bondsep.shape
    assert p_gold.block_type_ind.shape == p_new.block_type_ind.shape


# def two_ubq_poses(ubq_res, torch_device):
#     p1 = PoseStackBuilder.one_structure_from_polymeric_residues(ubq_res[:40], torch_device)
#     p2 = PoseStackBuilder.one_structure_from_polymeric_residues(ubq_res[:60], torch_device)
#     return PoseStackBuilder.from_poses([p1, p2], torch_device)


def test_pose_stack_builder_create_inter_residue_connections(ubq_res, torch_device):
    connections_by_name = find_simple_polymeric_connections(ubq_res[:4])
    inter_residue_connections = PoseStackBuilder._create_inter_residue_connections(
        ubq_res[:4], connections_by_name, torch_device
    )

    assert inter_residue_connections.shape == (1, 4, 2, 2)
    assert inter_residue_connections.device == torch_device

    assert inter_residue_connections[0, 0, 0, 0] == -1
    assert inter_residue_connections[0, 0, 0, 1] == -1

    assert inter_residue_connections[0, 0, 1, 0] == 1
    assert inter_residue_connections[0, 0, 1, 1] == 0
    assert inter_residue_connections[0, 1, 0, 0] == 0
    assert inter_residue_connections[0, 1, 0, 1] == 1

    assert inter_residue_connections[0, 1, 1, 0] == 2
    assert inter_residue_connections[0, 1, 1, 1] == 0
    assert inter_residue_connections[0, 2, 0, 0] == 1
    assert inter_residue_connections[0, 2, 0, 1] == 1

    assert inter_residue_connections[0, 2, 1, 0] == 3
    assert inter_residue_connections[0, 2, 1, 1] == 0
    assert inter_residue_connections[0, 3, 0, 0] == 2
    assert inter_residue_connections[0, 3, 0, 1] == 1

    assert inter_residue_connections[0, 3, 1, 0] == -1
    assert inter_residue_connections[0, 3, 1, 1] == -1


def test_pose_stack_builder_resolve_bond_separation(ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res[1:4])
    bonds = PoseStackBuilder._determine_single_structure_inter_block_bondsep(
        ubq_res[1:4], connections, torch_device
    )
    assert bonds[0, 0, 1, 1, 0] == 1
    assert bonds[0, 1, 2, 1, 0] == 1
    assert bonds[0, 1, 0, 0, 1] == 1
    assert bonds[0, 2, 1, 0, 1] == 1
    assert bonds[0, 0, 2, 1, 0] == 4
    assert bonds[0, 2, 0, 0, 1] == 4


def test_concatenate_pose_stacks_ctor(ubq_res, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:40], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:60], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    assert poses.block_type_ind.shape == (2, 60)
    assert poses.coords.shape == (2, 959, 3)
    assert poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)
