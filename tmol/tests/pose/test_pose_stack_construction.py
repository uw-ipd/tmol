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

    # assert p.residue_coords.shape == (1, 1228, 3)
    # assert p.coords.shape == p.residue_coords.shape
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

    # numpy.testing.assert_allclose(
    #     res_coords_gold, p.residue_coords[0], atol=1e-5, rtol=1e-5
    # )
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

    # assert p_gold.residue_coords.shape == p_new.residue_coords.shape
    assert p_gold.coords.shape == p_new.coords.shape
    assert (
        p_gold.inter_residue_connections.shape == p_new.inter_residue_connections.shape
    )
    assert p_gold.inter_block_bondsep.shape == p_new.inter_block_bondsep.shape
    assert p_gold.block_type_ind.shape == p_new.block_type_ind.shape


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


def test_create_pose_from_sequence(fresh_default_packed_block_types, torch_device):
    pbt = fresh_default_packed_block_types
    seqs = [["A", "P", "L", "F"], ["F", "P", "D"], ["A", "S", "F"]]
    PoseStackBuilder.pose_stack_from_monomer_polymer_sequences(pbt, seqs)


def test_pose_stack_builder_find_inter_block_sep_for_polymeric_monomers_lcaa(
    torch_device,
):

    # lets's conceive of a set of three bts, all w/ lcaa-like backbones
    def i64(x):
        return torch.tensor(x, dtype=torch.int64, device=torch_device)

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=torch_device)

    bt_polymeric_down_to_up_nbonds = i32([2, 2, 2])
    bt_up_conn_inds = i32([1, 1, 1])
    bt_down_conn_inds = i32([0, 0, 0])
    n_chains = 1
    max_n_res = 4
    max_n_conn = 2
    real_res = torch.tensor(
        [[True, True, True, True]], dtype=torch.bool, device=torch_device
    )
    block_type_ind64 = i64([[1, 2, 0, 1]])

    inter_block_separation64 = (
        PoseStackBuilder._find_inter_block_separation_for_polymeric_monomers_heavy(
            torch_device,
            bt_polymeric_down_to_up_nbonds,
            bt_up_conn_inds,
            bt_down_conn_inds,
            n_chains,
            max_n_res,
            max_n_conn,
            real_res,
            block_type_ind64,
        )
    )

    gold_inter_block_separation64 = i64(
        [
            [
                [
                    [[0, 2], [2, 0]],  # 0, 0
                    [[3, 5], [1, 3]],  # 0, 1
                    [[6, 8], [4, 6]],  # 0, 2
                    [[9, 11], [7, 9]],  # 0, 3
                ],
                [
                    [[3, 1], [5, 3]],  # 1, 0
                    [[0, 2], [2, 0]],  # 1, 1
                    [[3, 5], [1, 3]],  # 1, 2
                    [[6, 8], [4, 6]],  # 1, 3
                ],
                [
                    [[6, 4], [8, 6]],  # 2, 0
                    [[3, 1], [5, 3]],  # 2, 1
                    [[0, 2], [2, 0]],  # 2, 2
                    [[3, 5], [1, 3]],  # 2, 3
                ],
                [
                    [[9, 7], [11, 9]],  # 3, 0
                    [[6, 4], [8, 6]],  # 3, 1
                    [[3, 1], [5, 3]],  # 3, 2
                    [[0, 2], [2, 0]],  # 3, 3
                ],
            ]
        ]
    )

    torch.testing.assert_close(gold_inter_block_separation64, inter_block_separation64)


def test_pose_stack_builder_find_inter_block_sep_for_polymeric_monomers_mix_alpha_and_beta(
    torch_device,
):
    # this time, mix alpha- and beta amino acids in a chain

    def i64(x):
        return torch.tensor(x, dtype=torch.int64, device=torch_device)

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=torch_device)

    bt_polymeric_down_to_up_nbonds = i32([2, 2, 2, 3, 3, 3])
    bt_up_conn_inds = i32([1, 1, 1, 1, 1, 1])
    bt_down_conn_inds = i32([0, 0, 0, 0, 0, 0])
    n_chains = 1
    max_n_res = 4
    max_n_conn = 2
    real_res = torch.tensor(
        [[True, True, True, True]], dtype=torch.bool, device=torch_device
    )
    block_type_ind64 = i64([[1, 2, 4, 1]])

    inter_block_separation64 = (
        PoseStackBuilder._find_inter_block_separation_for_polymeric_monomers_heavy(
            torch_device,
            bt_polymeric_down_to_up_nbonds,
            bt_up_conn_inds,
            bt_down_conn_inds,
            n_chains,
            max_n_res,
            max_n_conn,
            real_res,
            block_type_ind64,
        )
    )

    gold_inter_block_separation64 = i64(
        [
            [
                [
                    [[0, 2], [2, 0]],  # 0, 0
                    [[3, 5], [1, 3]],  # 0, 1
                    [[6, 9], [4, 7]],  # 0, 2
                    [[10, 12], [8, 10]],  # 0, 3
                ],
                [
                    [[3, 1], [5, 3]],  # 1, 0
                    [[0, 2], [2, 0]],  # 1, 1
                    [[3, 6], [1, 4]],  # 1, 2
                    [[7, 9], [5, 7]],  # 1, 3
                ],
                [
                    [[6, 4], [9, 7]],  # 2, 0
                    [[3, 1], [6, 4]],  # 2, 1
                    [[0, 3], [3, 0]],  # 2, 2
                    [[4, 6], [1, 3]],  # 2, 3
                ],
                [
                    [[10, 8], [12, 10]],  # 3, 0
                    [[7, 5], [9, 7]],  # 3, 1
                    [[4, 1], [6, 3]],  # 3, 2
                    [[0, 2], [2, 0]],  # 3, 3
                ],
            ]
        ]
    )

    # print("gold_inter_block_separation64")
    # print(gold_inter_block_separation64)
    # print("inter_block_separation64")
    # print(inter_block_separation64)

    torch.testing.assert_close(gold_inter_block_separation64, inter_block_separation64)
