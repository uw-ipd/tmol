import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION
from tmol.io import pose_stack_from_pdb

from tmol.pose import InterBlockBondsep, PoseStackBuilder


def test_concatenate_pose_stacks_ctor(ubq_pdb, default_database, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=40)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=60)
    poses = PoseStackBuilder.from_poses([p1, p2], torch.device(torch_device.type))
    assert poses.block_type_ind.shape == (2, 60)
    assert poses.coords.shape == (2, 962, 3)  # fd 959->961 for nterm
    max_n_conn = poses.packed_block_types.max_n_conn
    assert poses.inter_block_bondsep.shape == (2, 60, 60, max_n_conn, max_n_conn)
    assert poses.device == torch_device
    torch.testing.assert_close(
        poses.block_ind_for_rot,
        torch.arange(60, dtype=torch.int32, device=torch_device).repeat(2),
    )


def test_create_pose_from_sequence(fresh_default_packed_block_types, torch_device):
    pbt = fresh_default_packed_block_types
    seqs = [["A", "P", "L", "F"], ["F", "P", "D"], ["A", "S", "F"]]
    chain_lengths = [[len(seq)] for seq in seqs]
    PoseStackBuilder.from_block_type_names(pbt, seqs, chain_lengths)


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


def test_blocks_without_connections_are_all_the_cap_apart(torch_device):
    counts = torch.zeros((2, 3), dtype=torch.int32, device=torch_device)
    intra = torch.full(
        (2, 3, 3, 3), MAX_SIG_BOND_SEPARATION, dtype=torch.int32, device=torch_device
    )
    connections = torch.full((2, 3, 3, 2), -1, dtype=torch.int64, device=torch_device)

    result = InterBlockBondsep.from_bonded_graph(counts, intra, connections)

    assert result.shape == (2, 3, 3, 3, 3)
    assert torch.all(result.to_dense() == MAX_SIG_BOND_SEPARATION)


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


def test_construct_pose_stack_containing_disulfides_smoke(
    fresh_default_packed_block_types, torch_device
):
    pbt = fresh_default_packed_block_types
    sequences = [
        ["ALA", "PRO", "CYD--dslf-first", "LEU", "CYD--dslf-first", "PHE"],
        ["PHE", "CYD--dslf-foo", "PRO", "CYD--dslf-foo", "ASP"],
    ]

    chain_lengths = [[len(seq)] for seq in sequences]
    _ = PoseStackBuilder.from_block_type_names(pbt, sequences, chain_lengths)


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


def test_from_block_type_names_smoke(
    fresh_default_packed_block_types, torch_device
):  # noqa: C901
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
    pose_stack = PoseStackBuilder.from_block_type_names(pbt, sequences, chain_lengths)

    # what is the inter_block_bondsep that should be computed?
    i_to_ip1_no_dslf_gold = torch.tensor(
        [[3, 5, 6], [1, 3, 6], [6, 6, 6]], dtype=torch.int32, device=torch_device
    )
    ibb_gold = torch.full(
        (n_poses, max_n_res, max_n_res, max_n_conn, max_n_conn),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int8,
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

    # connection slots past the three these residue types have are padding
    ibb = pose_stack.inter_block_bondsep.to_dense()
    torch.testing.assert_close(ibb[..., :max_n_conn, :max_n_conn], ibb_gold)
    padding = torch.ones_like(ibb, dtype=torch.bool)
    padding[..., :max_n_conn, :max_n_conn] = False
    assert bool((ibb[padding] == MAX_SIG_BOND_SEPARATION).all())
