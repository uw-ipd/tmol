import numpy
import torch

from tmol.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_pose_stack_builder_connection_ctor(ubq_res, default_database, torch_device):
    connections = find_simple_polymeric_connections(ubq_res)
    p = PoseStackBuilder.one_structure_from_residues_and_connections(
        default_database.chemical, ubq_res, connections, torch_device
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
    ubq_res, default_database, torch_device
):
    connections = find_simple_polymeric_connections(ubq_res)
    p_gold = PoseStackBuilder.one_structure_from_residues_and_connections(
        default_database.chemical, ubq_res, connections, torch_device
    )
    p_new = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res, torch_device
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

    assert inter_residue_connections[0, 0, 0, 0] == 1  # nterm, connection 0 is "up"
    assert inter_residue_connections[0, 0, 0, 1] == 0
    assert inter_residue_connections[0, 0, 1, 0] == -1
    assert inter_residue_connections[0, 0, 1, 1] == -1

    assert inter_residue_connections[0, 1, 0, 0] == 0
    assert inter_residue_connections[0, 1, 0, 1] == 0
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


def test_concatenate_pose_stacks_ctor(ubq_res, default_database, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:40], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:60], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    assert poses.block_type_ind.shape == (2, 60)
    assert poses.coords.shape == (2, 961, 3)  # fd 959->961 for nterm
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

    ibs64 = PoseStackBuilder._find_inter_block_separation_for_polymeric_monomers_heavy(
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
    inter_block_separation64 = ibs64

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


def test_pose_stack_builder_inter_block_sep_mix_alpha_and_beta(
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

    ibs64 = PoseStackBuilder._find_inter_block_separation_for_polymeric_monomers_heavy(
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
    inter_block_separation64 = ibs64

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


def test_take_real_conn_conn_intrablock_pairs(
    fresh_default_packed_block_types, torch_device
):
    pass


def test_take_real_conn_conn_intrablock_pairs_heavy(torch_device):
    ala_bt = 0
    cyd_bt = 1
    n_bt = 2
    max_n_conn = 3
    pbt_n_conn = torch.tensor([2, 3], dtype=torch.int32, device=torch_device)
    pbt_conn_at_intrablock_bond_sep = torch.full(
        (n_bt, max_n_conn, max_n_conn),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int32,
        device=torch_device,
    )
    # self, ala
    pbt_conn_at_intrablock_bond_sep[0, 0, 0] = 0
    pbt_conn_at_intrablock_bond_sep[0, 1, 1] = 0

    # other, ala
    pbt_conn_at_intrablock_bond_sep[0, 0, 1] = 2
    pbt_conn_at_intrablock_bond_sep[0, 1, 0] = 2

    # self, cyd
    pbt_conn_at_intrablock_bond_sep[1, 0, 0] = 0
    pbt_conn_at_intrablock_bond_sep[1, 1, 1] = 0
    pbt_conn_at_intrablock_bond_sep[1, 2, 2] = 0

    # other, cyd
    pbt_conn_at_intrablock_bond_sep[1, 0, 1] = 2
    pbt_conn_at_intrablock_bond_sep[1, 1, 0] = 2
    pbt_conn_at_intrablock_bond_sep[1, 0, 2] = 3
    pbt_conn_at_intrablock_bond_sep[1, 2, 0] = 3
    pbt_conn_at_intrablock_bond_sep[1, 1, 2] = 3
    pbt_conn_at_intrablock_bond_sep[1, 2, 1] = 3

    block_types64 = torch.tensor(
        [[ala_bt, ala_bt, cyd_bt, ala_bt], [ala_bt, ala_bt, -1, -1]],
        dtype=torch.int64,
        device=torch_device,
    )
    real_blocks = block_types64 != -1

    pconn_matrix, *_ = PoseStackBuilder._take_real_conn_conn_intrablock_pairs_heavy(
        pbt_n_conn, pbt_conn_at_intrablock_bond_sep, block_types64, real_blocks
    )

    pconn_matrix_gold = torch.full(
        (2, 2 + 2 + 3 + 2, 2 + 2 + 3 + 2),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int32,
        device=torch_device,
    )

    pconn_matrix_gold[0, 0:2, 0:2] = 2
    pconn_matrix_gold[0, 0, 0] = 0
    pconn_matrix_gold[0, 1, 1] = 0

    pconn_matrix_gold[0, 2:4, 2:4] = 2
    pconn_matrix_gold[0, 2, 2] = 0
    pconn_matrix_gold[0, 3, 3] = 0

    pconn_matrix_gold[0, 4:6, 4:6] = 2
    pconn_matrix_gold[0, 4, 4] = 0
    pconn_matrix_gold[0, 5, 5] = 0
    pconn_matrix_gold[0, 4:6, 6] = 3
    pconn_matrix_gold[0, 6, 4:6] = 3
    pconn_matrix_gold[0, 6, 6] = 0

    pconn_matrix_gold[0, 7:9, 7:9] = 2
    pconn_matrix_gold[0, 7, 7] = 0
    pconn_matrix_gold[0, 8, 8] = 0

    pconn_matrix_gold[1, 0:2, 0:2] = 2
    pconn_matrix_gold[1, 0, 0] = 0
    pconn_matrix_gold[1, 1, 1] = 0

    pconn_matrix_gold[1, 2:4, 2:4] = 2
    pconn_matrix_gold[1, 2, 2] = 0
    pconn_matrix_gold[1, 3, 3] = 0

    torch.testing.assert_close(pconn_matrix_gold, pconn_matrix)


def test_find_connection_pairs_for_residue_subset(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    ala_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "ALA")
    cyd_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "CYD")

    sequences = [["ALA", "ALA", "CYD", "ALA", "CYD"], ["ALA", "CYD", "ALA", "CYD"]]
    block_types = torch.tensor(
        [
            [ala_bt, ala_bt, cyd_bt, ala_bt, cyd_bt],
            [ala_bt, cyd_bt, ala_bt, cyd_bt, -1],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    residue_connections = [[(2, "dslf", 4, "dslf")], [(1, "dslf", 3, "dslf")]]

    ps_conns = PoseStackBuilder._find_connection_pairs_for_residue_subset(
        pbt, sequences, block_types, residue_connections
    )

    ps_conns_gold = [[(2, 2, 4, 2)], [(1, 2, 3, 2)]]
    assert len(ps_conns) == len(ps_conns_gold)
    for p_conn, p_conn_gold in zip(ps_conns, ps_conns_gold):
        assert len(p_conn) == len(p_conn_gold)
        for conn, conn_gold in zip(p_conn, p_conn_gold):
            assert conn == conn_gold


def test_find_connection_pairs_for_residue_subset2(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    abt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "ALA")
    cbt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "CYD")
    a = "ALA"
    c = "CYD"

    sequences = [[a, a, c, a, c, a, c, a, c, a, a], [a, c, a, c]]
    block_types = torch.tensor(
        [
            [abt, abt, cbt, abt, cbt, abt, cbt, abt, cbt, abt, abt],
            [abt, cbt, abt, cbt, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    residue_connections = [
        [(2, "dslf", 8, "dslf"), (4, "dslf", 6, "dslf")],
        [(1, "dslf", 3, "dslf")],
    ]

    ps_conns = PoseStackBuilder._find_connection_pairs_for_residue_subset(
        pbt, sequences, block_types, residue_connections
    )

    ps_conns_gold = [[(2, 2, 8, 2), (4, 2, 6, 2)], [(1, 2, 3, 2)]]
    assert len(ps_conns) == len(ps_conns_gold)
    for p_conn, p_conn_gold in zip(ps_conns, ps_conns_gold):
        assert len(p_conn) == len(p_conn_gold)
        for conn, conn_gold in zip(p_conn, p_conn_gold):
            assert conn == conn_gold


def test_find_connections_in_sequences(fresh_default_packed_block_types, torch_device):
    pbt = fresh_default_packed_block_types
    sequences = [
        ["ALA", "ALA", "CYD--dslf-first", "ALA", "CYD--dslf-first"],
        ["ALA", "CYD--dslf-foo", "ALA", "CYD--dslf-foo"],
    ]
    trimmed_seqs, ps_conns = PoseStackBuilder._find_connections_in_sequences(
        pbt, sequences
    )

    trimmed_seqs_gold = [
        ["ALA", "ALA", "CYD", "ALA", "CYD"],
        ["ALA", "CYD", "ALA", "CYD"],
    ]
    ps_conns_gold = [[(2, "dslf", 4, "dslf")], [(1, "dslf", 3, "dslf")]]

    assert trimmed_seqs == trimmed_seqs_gold
    assert ps_conns == ps_conns_gold


def test_find_connection_pairs_for_residue_subset_w_errors1(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    ala_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "ALA")
    cyd_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "CYD")

    sequences = [["ALA", "ALA", "CYD", "ALA", "CYD"], ["ALA", "CYD", "ALA", "CYD"]]
    block_types = torch.tensor(
        [
            [ala_bt, ala_bt, cyd_bt, ala_bt, cyd_bt],
            [ala_bt, cyd_bt, ala_bt, cyd_bt, -1],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    residue_connections = [[(2, "bslf", 4, "dslf")], [(1, "dslf", 3, "dslf")]]

    succeeded = False
    try:
        _ = PoseStackBuilder._find_connection_pairs_for_residue_subset(
            pbt, sequences, block_types, residue_connections
        )
        succeeded = True
    except ValueError as e:
        assert str(e) == (
            "Failed to find connection 'bslf' on residue type 'CYD' which "
            + "is listed as forming a chemical bond"
            + " to connection 'dslf' on residue type 'CYD'\n"
            + "Valid connection names on 'CYD' are: 'down', 'up', 'dslf'"
        )
    assert not succeeded


def test_find_connection_pairs_for_residue_subset_w_errors2(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    ala_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "ALA")
    cyd_bt = next(i for i, bt in enumerate(pbt.active_block_types) if bt.name == "CYD")

    sequences = [["ALA", "ALA", "CYD", "ALA", "CYD"], ["ALA", "CYD", "ALA", "CYD"]]
    block_types = torch.tensor(
        [
            [ala_bt, ala_bt, cyd_bt, ala_bt, cyd_bt],
            [ala_bt, cyd_bt, ala_bt, cyd_bt, -1],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    residue_connections = [[(2, "dslf", 4, "gslf")], [(1, "dslf", 3, "dslf")]]

    succeeded = False
    try:
        _ = PoseStackBuilder._find_connection_pairs_for_residue_subset(
            pbt, sequences, block_types, residue_connections
        )
        succeeded = True
    except ValueError as e:
        assert str(e) == (
            "Failed to find connection 'gslf' on residue type 'CYD'"
            + " which is listed as forming a chemical bond"
            + " to connection 'dslf' on residue type 'CYD'\n"
            + "Valid connection names on 'CYD' are: 'down', 'up', 'dslf'"
        )
    assert not succeeded


def test_calculate_interblock_bondsep_from_connectivity_graph_heavy(torch_device):
    pbt_max_n_conn = 3
    block_n_conn = torch.tensor(
        [[2, 2, 3, 2, 3], [2, 3, 2, 3, 0]], dtype=torch.int32, device=torch_device
    )
    pose_n_pconn = torch.tensor([12, 10], dtype=torch.int32, device=torch_device)
    pconn_matrix = torch.tensor(
        [
            [
                [0, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # ala down
                [2, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # ala up
                [6, 1, 0, 2, 6, 6, 6, 6, 6, 6, 6, 6],  # ala down
                [6, 6, 2, 0, 1, 6, 6, 6, 6, 6, 6, 6],  # ala up
                [6, 6, 6, 1, 0, 2, 3, 6, 6, 6, 6, 6],  # cyd down
                [6, 6, 6, 6, 2, 0, 3, 1, 6, 6, 6, 6],  # cyd up
                [6, 6, 6, 6, 3, 3, 0, 6, 6, 6, 6, 1],  # cyd dslf
                [6, 6, 6, 6, 6, 1, 6, 0, 2, 6, 6, 6],  # ala down
                [6, 6, 6, 6, 6, 6, 6, 2, 0, 1, 6, 6],  # ala up
                [6, 6, 6, 6, 6, 6, 6, 6, 1, 0, 2, 3],  # cyd down
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 0, 3],  # cyd up
                [6, 6, 6, 6, 6, 6, 1, 6, 6, 3, 3, 0],  # cyd dslf
            ],
            [
                [0, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # ala down
                [2, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # ala up
                [6, 1, 0, 2, 3, 6, 6, 6, 6, 6, 6, 6],  # cyd down
                [6, 6, 2, 0, 3, 1, 6, 6, 6, 6, 6, 6],  # cyd up
                [6, 6, 3, 3, 0, 6, 6, 6, 6, 1, 6, 6],  # cyd dslf
                [6, 6, 6, 1, 6, 0, 2, 6, 6, 6, 6, 6],  # ala down
                [6, 6, 6, 6, 6, 2, 0, 1, 6, 6, 6, 6],  # ala up
                [6, 6, 6, 6, 6, 6, 1, 0, 2, 3, 6, 6],  # cyd down
                [6, 6, 6, 6, 6, 6, 6, 2, 0, 3, 6, 6],  # cyd up
                [6, 6, 6, 6, 1, 6, 6, 3, 3, 0, 6, 6],  # cyd dslf
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # empty
                [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # empty
            ],
        ],
        dtype=torch.int32,
        device=torch_device,
    )

    ibb = PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph_heavy(
        pbt_max_n_conn, torch_device, block_n_conn, pose_n_pconn, pconn_matrix
    )
    inter_block_bondsep = ibb

    inter_block_bondsep_gold = torch.tensor(
        [
            [
                [
                    [[0, 2, 6], [2, 0, 6], [6, 6, 6]],
                    [[3, 5, 6], [1, 3, 6], [6, 6, 6]],
                    [[6, 6, 6], [4, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
                [
                    [[3, 1, 6], [5, 3, 6], [6, 6, 6]],
                    [[0, 2, 6], [2, 0, 6], [6, 6, 6]],
                    [[3, 5, 6], [1, 3, 4], [6, 6, 6]],
                    [[6, 6, 6], [4, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 5], [6, 6, 6]],
                ],
                [
                    [[6, 4, 6], [6, 6, 6], [6, 6, 6]],
                    [[3, 1, 6], [5, 3, 6], [6, 4, 6]],
                    [[0, 2, 3], [2, 0, 3], [3, 3, 0]],
                    [[3, 5, 6], [1, 3, 6], [4, 5, 6]],
                    [[6, 6, 4], [4, 6, 4], [4, 4, 1]],
                ],
                [
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 4, 6], [6, 6, 6], [6, 6, 6]],
                    [[3, 1, 4], [5, 3, 5], [6, 6, 6]],
                    [[0, 2, 6], [2, 0, 6], [6, 6, 6]],
                    [[3, 5, 5], [1, 3, 4], [6, 6, 6]],
                ],
                [
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 5, 6]],
                    [[6, 4, 4], [6, 6, 4], [4, 4, 1]],
                    [[3, 1, 6], [5, 3, 6], [5, 4, 6]],
                    [[0, 2, 3], [2, 0, 3], [3, 3, 0]],
                ],
            ],
            [
                [
                    [[0, 2, 6], [2, 0, 6], [6, 6, 6]],
                    [[3, 5, 6], [1, 3, 4], [6, 6, 6]],
                    [[6, 6, 6], [4, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 5], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
                [
                    [[3, 1, 6], [5, 3, 6], [6, 4, 6]],
                    [[0, 2, 3], [2, 0, 3], [3, 3, 0]],
                    [[3, 5, 6], [1, 3, 6], [4, 5, 6]],
                    [[6, 6, 4], [4, 6, 4], [4, 4, 1]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
                [
                    [[6, 4, 6], [6, 6, 6], [6, 6, 6]],
                    [[3, 1, 4], [5, 3, 5], [6, 6, 6]],
                    [[0, 2, 6], [2, 0, 6], [6, 6, 6]],
                    [[3, 5, 5], [1, 3, 4], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
                [
                    [[6, 6, 6], [6, 6, 6], [6, 5, 6]],
                    [[6, 4, 4], [6, 6, 4], [4, 4, 1]],
                    [[3, 1, 6], [5, 3, 6], [5, 4, 6]],
                    [[0, 2, 3], [2, 0, 3], [3, 3, 0]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
                [
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                    [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                ],
            ],
        ],
        dtype=torch.int32,
        device=torch_device,
    )

    # numpy.set_printoptions(threshold=10000)
    # print("inter_block_bondsep")
    # print(inter_block_bondsep.cpu().numpy())

    torch.testing.assert_close(inter_block_bondsep, inter_block_bondsep_gold)


def test_incorporate_extra_connections_into_inter_res_conn_set(torch_device):
    n_poses, max_n_blocks, max_n_conn = 2, 5, 3
    resolved_expoly_connections = [[(2, 2, 4, 2)], [(1, 2, 3, 2)]]
    inter_residue_connections64 = torch.full(
        (n_poses, max_n_blocks, max_n_conn, 2),
        -1,
        dtype=torch.int64,
        device=torch_device,
    )
    inter_residue_connections64_gold = inter_residue_connections64.clone()

    PoseStackBuilder._incorporate_extra_connections_into_inter_res_conn_set(
        resolved_expoly_connections, inter_residue_connections64
    )

    inter_residue_connections64_gold[0, 2, 2, 0] = 4
    inter_residue_connections64_gold[0, 2, 2, 1] = 2
    inter_residue_connections64_gold[0, 4, 2, 0] = 2
    inter_residue_connections64_gold[0, 4, 2, 1] = 2

    inter_residue_connections64_gold[1, 1, 2, 0] = 3
    inter_residue_connections64_gold[1, 1, 2, 1] = 2
    inter_residue_connections64_gold[1, 3, 2, 0] = 1
    inter_residue_connections64_gold[1, 3, 2, 1] = 2

    torch.testing.assert_close(
        inter_residue_connections64_gold, inter_residue_connections64
    )


def test_incorporate_extra_connections_into_inter_res_conn_set2(torch_device):
    n_poses, max_n_blocks, max_n_conn = 2, 6, 3
    resolved_expoly_connections = [[(2, 2, 4, 2), (1, 2, 5, 2)], [(1, 2, 3, 2)]]
    inter_residue_connections64 = torch.full(
        (n_poses, max_n_blocks, max_n_conn, 2),
        -1,
        dtype=torch.int64,
        device=torch_device,
    )
    inter_residue_connections64_gold = inter_residue_connections64.clone()

    PoseStackBuilder._incorporate_extra_connections_into_inter_res_conn_set(
        resolved_expoly_connections, inter_residue_connections64
    )

    inter_residue_connections64_gold[0, 2, 2, 0] = 4
    inter_residue_connections64_gold[0, 2, 2, 1] = 2
    inter_residue_connections64_gold[0, 4, 2, 0] = 2
    inter_residue_connections64_gold[0, 4, 2, 1] = 2

    inter_residue_connections64_gold[0, 1, 2, 0] = 5
    inter_residue_connections64_gold[0, 1, 2, 1] = 2
    inter_residue_connections64_gold[0, 5, 2, 0] = 1
    inter_residue_connections64_gold[0, 5, 2, 1] = 2

    inter_residue_connections64_gold[1, 1, 2, 0] = 3
    inter_residue_connections64_gold[1, 1, 2, 1] = 2
    inter_residue_connections64_gold[1, 3, 2, 0] = 1
    inter_residue_connections64_gold[1, 3, 2, 1] = 2

    torch.testing.assert_close(
        inter_residue_connections64_gold, inter_residue_connections64
    )


def test_incorporate_inter_residue_connections_into_connectivity_graph(torch_device):
    n_poses, max_n_blocks, max_n_conn, max_n_pconn = 2, 5, 3, 12

    inter_residue_connections64 = torch.full(
        (n_poses, max_n_blocks, max_n_conn, 2),
        -1,
        dtype=torch.int64,
        device=torch_device,
    )

    def connect_up_down_pair(pind, r1ind, r2ind):
        inter_residue_connections64[pind, r1ind, 1, 0] = r2ind
        inter_residue_connections64[pind, r2ind, 0, 0] = r1ind
        inter_residue_connections64[pind, r1ind, 1, 1] = 0
        inter_residue_connections64[pind, r1ind, 0, 0] = 1

    connect_up_down_pair(0, 0, 1)
    connect_up_down_pair(0, 1, 2)
    connect_up_down_pair(0, 2, 3)
    connect_up_down_pair(0, 3, 4)
    connect_up_down_pair(1, 0, 1)
    connect_up_down_pair(1, 1, 2)
    connect_up_down_pair(1, 2, 3)

    # imagined sequence: [[ala, ala, cyd, ala, cyd], [ala, cyd, ala, cyd]]
    resolved_expoly_connections = [[(2, 2, 4, 2)], [(1, 2, 3, 2)]]

    PoseStackBuilder._incorporate_extra_connections_into_inter_res_conn_set(
        resolved_expoly_connections, inter_residue_connections64
    )

    pconn_offset = torch.tensor(
        [[0, 2, 4, 7, 9], [0, 2, 5, 7, 7]], dtype=torch.int64, device=torch_device
    )

    pconn_matrix = torch.full(
        (n_poses, max_n_pconn, max_n_pconn),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int32,
        device=torch_device,
    )
    pconn_matrix_gold = pconn_matrix.clone()
    for i in range(n_poses):
        for j in range(max_n_blocks):
            for k in range(max_n_conn):
                if inter_residue_connections64[i, j, k, 0] != -1:
                    partner = inter_residue_connections64[i, j, k, 0]
                    j_offset = pconn_offset[i, j]
                    partner_offset = pconn_offset[i, partner]
                    partner_conn = inter_residue_connections64[i, j, k, 1]
                    pconn_matrix_gold[
                        i, j_offset + k, partner_offset + partner_conn
                    ] = 1

    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        inter_residue_connections64, pconn_offset, pconn_matrix
    )

    torch.testing.assert_close(pconn_matrix_gold, pconn_matrix)


def test_construct_pose_stack_containing_disulfides_smoke(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    sequences = [
        ["ALA", "PRO", "CYD--dslf-first", "LEU", "CYD--dslf-first", "PHE"],
        ["PHE", "CYD--dslf-foo", "PRO", "CYD--dslf-foo", "ASP"],
    ]

    _ = PoseStackBuilder.pose_stack_from_monomer_sequences_w_extrapolymeric_conns(
        pbt, sequences
    )


def interblock_dslf_self_correction(ibb, res_bound_to_next, p, i):
    """Set the inter-block-bondsep for pose p, res i"""
    ibb[p, i, i, :, 2] = 3
    ibb[p, i, i, 2, :] = 3
    ibb[p, i, i, 2, 2] = 0
    if res_bound_to_next[p, i - 1]:
        ibb[p, i - 1, i, 1, 2] = 4
        ibb[p, i, i - 1, 2, 1] = 4
    if res_bound_to_next[p, i]:
        ibb[p, i, i + 1, 2, 0] = 4
        ibb[p, i + 1, i, 0, 2] = 4


def interblock_dslf_pair_correction(ibb, res_bound_to_next, p, i, j):
    ibb[p, i, j, :, 2] = 4
    ibb[p, i, j, 2, :] = 4
    ibb[p, j, i, :, 2] = 4
    ibb[p, j, i, 2, :] = 4
    ibb[p, i, j, 2, 2] = 1
    ibb[p, j, i, 2, 2] = 1
    if res_bound_to_next[p, i - 1]:
        ibb[p, i - 1, j, 1, 2] = 5
        ibb[p, j, i - 1, 2, 1] = 5
    if res_bound_to_next[p, i]:
        ibb[p, i + 1, j, 0, 2] = 5
        ibb[p, j, i + 1, 2, 0] = 5
    if res_bound_to_next[p, j - 1]:
        ibb[p, j - 1, i, 1, 2] = 5
        ibb[p, i, j - 1, 2, 1] = 5
    if res_bound_to_next[p, j]:
        ibb[p, j + 1, i, 0, 2] = 5
        ibb[p, i, j + 1, 2, 0] = 5


def test_pose_stack_from_sequences_smoke(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    n_poses, max_n_res, max_n_conn = 2, 8, 3
    sequences = [
        ["ALA", "PRO", "CYD--dslf-first", "LEU", "CYD--dslf-first", "PHE"],
        ["PHE", "GLY", "SER", "CYD--dslf-foo", "PRO", "CYD--dslf-foo", "ASP", "GLY"],
    ]
    chain_lengths = [[2, 4], [3, 3, 2]]
    # derived data
    has_dslf = [
        [True if resname.find("dslf") != -1 else False for resname in seq]
        for seq in sequences
    ]
    res_bound_to_next = torch.full(
        (n_poses, max_n_res), False, dtype=torch.bool
    )  # leave on cpu
    for i, lens in enumerate(chain_lengths):
        count = 0
        for j in lens:
            res_bound_to_next[i, count : (count + j - 1)] = True
            count += j

    # the call we are testing
    pose_stack = PoseStackBuilder.pose_stack_from_sequences(
        pbt, sequences, chain_lengths
    )

    # what is the inter_block_bondsep that should be computed?
    i_to_ip1_no_dslf_gold = torch.tensor(
        [[3, 5, 6], [1, 3, 6], [6, 6, 6]], dtype=torch.int32, device=torch_device
    )
    ibb_gold = torch.full(
        (n_poses, max_n_res, max_n_res, max_n_conn, max_n_conn),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int32,
        device=torch_device,
    )

    def fill_i_to_ip1_gold(p, i):
        ibb_gold[p, i, i + 1] = i_to_ip1_no_dslf_gold
        ibb_gold[p, i + 1, i] = torch.transpose(i_to_ip1_no_dslf_gold, 0, 1)
        if i - 2 >= 0 and res_bound_to_next[p, i - 2] and res_bound_to_next[p, i - 1]:
            ibb_gold[p, i - 2, i, 1, 0] = 4
            ibb_gold[p, i, i - 2, 0, 1] = 4
        if (
            i + 2 < max_n_res
            and res_bound_to_next[p, i]
            and res_bound_to_next[p, i + 1]
        ):
            ibb_gold[p, i + 2, i, 0, 1] = 4
            ibb_gold[p, i, i + 2, 1, 0] = 4

    for i in range(n_poses):
        for j in range(max_n_res):
            if res_bound_to_next[i, j]:
                fill_i_to_ip1_gold(i, j)

    i_self_no_dslf_gold = torch.tensor(
        [[0, 2, 6], [2, 0, 6], [6, 6, 6]], dtype=torch.int32, device=torch_device
    )

    def set_self_nodslf(p, i):
        ibb_gold[p, i, i] = i_self_no_dslf_gold

    for i in range(6):
        set_self_nodslf(0, i)
    for i in range(8):
        set_self_nodslf(1, i)

    for i, i_has_dslf in enumerate(has_dslf):
        for j, ij_has_dslf in enumerate(i_has_dslf):
            if ij_has_dslf:
                interblock_dslf_self_correction(ibb_gold, res_bound_to_next, i, j)

    # fix the bond separation distances to the residues on the other side of the
    # disfulide bond and the residues up and down the chain.
    interblock_dslf_pair_correction(ibb_gold, res_bound_to_next, 0, 2, 4)
    interblock_dslf_pair_correction(ibb_gold, res_bound_to_next, 1, 3, 5)

    torch.testing.assert_close(pose_stack.inter_block_bondsep, ibb_gold)
