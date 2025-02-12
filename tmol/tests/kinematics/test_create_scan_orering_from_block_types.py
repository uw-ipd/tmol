import torch

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)

from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

from tmol.kinematics.fold_forest import EdgeType
from tmol.kinematics.scan_ordering import (
    construct_kin_module_data_for_pose,
    _annotate_block_type_with_gen_scan_path_segs,
    _annotate_packed_block_type_with_gen_scan_path_segs,
)
from tmol.kinematics.compiled import inverse_kin, forward_kin_op


def test_gen_seg_scan_paths_block_type_annotation_smoke(fresh_default_restype_set):
    bt_list = [bt for bt in fresh_default_restype_set.residue_types if bt.name == "LEU"]
    for bt in bt_list:
        _annotate_block_type_with_gen_scan_path_segs(bt)
        assert hasattr(bt, "gen_seg_scan_path_segs")


def test_calculate_ff_edge_delays_for_two_res_ubq(ubq_pdb, torch_device):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=1, residue_end=3
    )

    res_not_connected = torch.zeros((1, 2, 2), dtype=torch.bool, device=torch_device)
    res_not_connected[0, 0, 0] = True  # simplest test case: not N-term
    res_not_connected[0, 1, 1] = True  # simplest test case: not C-term
    pose_stack = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, res_not_connected=res_not_connected
    )
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    max_n_edges = 2
    ff_edges = torch.zeros(
        (pose_stack.n_poses, max_n_edges, 4),
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 0
    ff_edges[0, 0, 2] = 1
    ff_edges[0, 1, 0] = EdgeType.root_jump
    ff_edges[0, 1, 1] = -1
    ff_edges[0, 1, 2] = 0
    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_edges,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )
    assert result is not None


def test_calculate_ff_edge_delays_for_6_res_ubq(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    torch_device = torch.device("cpu")
    # device = torch_device

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=1, residue_end=7
    )

    res_not_connected = torch.zeros((1, 6, 2), dtype=torch.bool, device=torch_device)
    res_not_connected[0, 0, 0] = True  # simplest test case: not N-term
    res_not_connected[0, 5, 1] = True  # simplest test case: not C-term
    pose_stack = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, res_not_connected=res_not_connected
    )
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    max_n_edges = 6
    ff_edges = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = EdgeType.jump
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4
    ff_edges[0, 2, 3] = 0

    ff_edges[0, 3, 0] = EdgeType.polymer
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = EdgeType.polymer
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    ff_edges[0, 5, 0] = EdgeType.root_jump
    ff_edges[0, 5, 1] = -1
    ff_edges[0, 5, 2] = 1

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_edges,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )

    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block_cpu,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = result

    assert dfs_order_of_ff_edges is not None
    assert n_ff_edges is not None
    assert ff_edge_parent is not None
    assert first_ff_edge_for_block_cpu is not None
    assert pose_stack_ff_parent is not None
    assert max_gen_depth_of_ff_edge is not None
    assert first_child_of_ff_edge is not None
    assert delay_for_edge is not None
    assert toposort_index_for_edge is not None


def test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_H(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    ff_edges = torch.tensor(ff_2ubq_6res_H)

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_edges,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )
    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block_cpu,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = result

    gold_dfs_order_of_ff_edges = torch.tensor(
        [[5, 2, 4, 3, 1, 0], [5, 4, 3, 2, 1, 0]], dtype=torch.int32
    )
    gold_n_ff_edges = torch.tensor([6, 6], dtype=torch.int32)
    gold_ff_edge_parent = torch.tensor(
        [[5, 5, 5, 2, 2, -1], [2, 2, 5, 5, 5, -1]], dtype=torch.int32
    )
    gold_first_ff_edge_for_block_cpu = torch.tensor(
        [[0, 5, 1, 3, 2, 4], [0, 2, 1, 3, 5, 4]], dtype=torch.int32
    )
    gold_pose_stack_ff_parent = torch.tensor(
        [[1, -1, 1, 4, 1, 4], [1, 4, 1, 4, -1, 4]], dtype=torch.int32
    )
    gold_max_gen_depth_of_ff_edge = torch.tensor(
        [[4, 4, 5, 4, 4, 5], [4, 4, 5, 4, 4, 5]], dtype=torch.int32
    )
    gold_first_child_of_ff_edge = torch.tensor(
        [[-1, -1, 3, -1, -1, 2], [-1, -1, 0, -1, -1, 2]], dtype=torch.int32
    )
    gold_delay_for_edge = torch.tensor(
        [[1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0]], dtype=torch.int32
    )
    gold_toposort_index_for_edge = torch.tensor(
        [6, 7, 1, 2, 8, 0, 5, 11, 4, 9, 10, 3], dtype=torch.int32
    )

    torch.testing.assert_close(gold_dfs_order_of_ff_edges, dfs_order_of_ff_edges)
    torch.testing.assert_close(gold_n_ff_edges, n_ff_edges)
    torch.testing.assert_close(gold_ff_edge_parent, ff_edge_parent)
    torch.testing.assert_close(
        gold_first_ff_edge_for_block_cpu, first_ff_edge_for_block_cpu
    )
    torch.testing.assert_close(gold_pose_stack_ff_parent, pose_stack_ff_parent)
    torch.testing.assert_close(gold_max_gen_depth_of_ff_edge, max_gen_depth_of_ff_edge)
    torch.testing.assert_close(gold_first_child_of_ff_edge, first_child_of_ff_edge)
    torch.testing.assert_close(gold_delay_for_edge, delay_for_edge)
    torch.testing.assert_close(gold_toposort_index_for_edge, toposort_index_for_edge)


def test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_U(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_U
):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs
    ff_2ubq_6res_U = torch.tensor(ff_2ubq_6res_U)

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_2ubq_6res_U,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )
    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block_cpu,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = result

    gold_dfs_order_of_ff_edges = torch.tensor(
        [[3, 1, 2, 0], [3, 2, 1, 0]], dtype=torch.int32
    )
    gold_n_ff_edges = torch.tensor([4, 4], dtype=torch.int32)
    gold_ff_edge_parent = torch.tensor(
        [[3, 3, 1, -1], [1, 3, 3, -1]], dtype=torch.int32
    )
    gold_first_ff_edge_for_block_cpu = torch.tensor(
        [[0, 0, 3, 2, 2, 1], [0, 0, 1, 2, 2, 3]], dtype=torch.int32
    )
    gold_pose_stack_ff_parent = torch.tensor(
        [[1, 2, -1, 4, 5, 2], [1, 2, 5, 4, 5, -1]], dtype=torch.int32
    )
    gold_max_gen_depth_of_ff_edge = torch.tensor(
        [[4, 4, 4, 5], [4, 4, 4, 5]], dtype=torch.int32
    )
    gold_first_child_of_ff_edge = torch.tensor(
        [[-1, 2, -1, 0], [-1, 0, -1, 1]], dtype=torch.int32
    )
    gold_delay_for_edge = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=torch.int32)
    gold_toposort_index_for_edge = torch.tensor(
        [1, 5, 6, 0, 4, 3, 7, 2], dtype=torch.int32
    )

    torch.testing.assert_close(gold_dfs_order_of_ff_edges, dfs_order_of_ff_edges)
    torch.testing.assert_close(gold_n_ff_edges, n_ff_edges)
    torch.testing.assert_close(gold_ff_edge_parent, ff_edge_parent)
    torch.testing.assert_close(
        gold_first_ff_edge_for_block_cpu, first_ff_edge_for_block_cpu
    )
    torch.testing.assert_close(gold_pose_stack_ff_parent, pose_stack_ff_parent)
    torch.testing.assert_close(gold_max_gen_depth_of_ff_edge, max_gen_depth_of_ff_edge)
    torch.testing.assert_close(gold_first_child_of_ff_edge, first_child_of_ff_edge)
    torch.testing.assert_close(gold_delay_for_edge, delay_for_edge)
    torch.testing.assert_close(gold_toposort_index_for_edge, toposort_index_for_edge)


def test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_K(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_K
):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs
    ff_2ubq_6res_K = torch.tensor(ff_2ubq_6res_K)

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_2ubq_6res_K,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )
    ff_2ubq_6res_K = torch.tensor(ff_2ubq_6res_K)

    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block_cpu,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = result

    gold_dfs_order_of_ff_edges = torch.tensor(
        [[5, 3, 4, 2, 1, 0], [5, 3, 4, 2, 1, 0]], dtype=torch.int32
    )
    gold_n_ff_edges = torch.tensor([6, 6], dtype=torch.int32)
    gold_ff_edge_parent = torch.tensor(
        [[5, 5, 5, 5, 3, -1], [5, 5, 5, 5, 3, -1]], dtype=torch.int32
    )
    gold_first_ff_edge_for_block_cpu = torch.tensor(
        [[0, 5, 1, 2, 3, 4], [4, 3, 2, 0, 5, 1]], dtype=torch.int32
    )
    gold_pose_stack_ff_parent = torch.tensor(
        [[1, -1, 1, 1, 1, 4], [1, 4, 4, 4, -1, 4]], dtype=torch.int32
    )
    gold_max_gen_depth_of_ff_edge = torch.tensor(
        [[4, 4, 4, 4, 4, 5], [4, 4, 4, 4, 4, 5]], dtype=torch.int32
    )
    gold_first_child_of_ff_edge = torch.tensor(
        [[-1, -1, -1, 4, -1, 0], [-1, -1, -1, 4, -1, 0]], dtype=torch.int32
    )
    gold_delay_for_edge = torch.tensor(
        [[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]], dtype=torch.int32
    )
    gold_toposort_index_for_edge = torch.tensor(
        [1, 4, 5, 6, 7, 0, 3, 8, 9, 10, 11, 2], dtype=torch.int32
    )

    torch.testing.assert_close(gold_dfs_order_of_ff_edges, dfs_order_of_ff_edges)
    torch.testing.assert_close(gold_n_ff_edges, n_ff_edges)
    torch.testing.assert_close(gold_ff_edge_parent, ff_edge_parent)
    torch.testing.assert_close(
        gold_first_ff_edge_for_block_cpu, first_ff_edge_for_block_cpu
    )
    torch.testing.assert_close(gold_pose_stack_ff_parent, pose_stack_ff_parent)
    torch.testing.assert_close(gold_max_gen_depth_of_ff_edge, max_gen_depth_of_ff_edge)
    torch.testing.assert_close(gold_first_child_of_ff_edge, first_child_of_ff_edge)
    torch.testing.assert_close(gold_delay_for_edge, delay_for_edge)
    torch.testing.assert_close(gold_toposort_index_for_edge, toposort_index_for_edge)


def test_calculate_parent_block_conn_in_and_out_for_two_copies_of_6_res_ubq(
    stack_of_two_six_res_ubqs_no_term, torch_device, ff_2ubq_6res_H
):
    from tmol.kinematics.compiled.compiled_ops import (
        calculate_ff_edge_delays,
        get_block_parent_connectivity_from_toposort,
    )

    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs
    ff_2ubq_6res_H = torch.tensor(ff_2ubq_6res_H)

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        ff_2ubq_6res_H,
        pbt_gssps.scan_path_seg_that_builds_output_conn,
        pbt_gssps.nodes_for_gen,
        pbt_gssps.scan_path_seg_starts,
    )
    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = tuple(x.to(device=torch_device) for x in result)
    pose_stack_block_in_and_first_out = get_block_parent_connectivity_from_toposort(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_2ubq_6res_H.to(device=torch_device),
        first_ff_edge_for_block,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
        pbt.n_conn,
        pbt.polymeric_conn_inds,
    )
    gold_pose_stack_block_in_and_first_out = torch.tensor(
        [
            [[1, 3], [3, 2], [0, 3], [1, 3], [2, 0], [0, 3]],
            [[1, 3], [2, 0], [0, 3], [1, 3], [3, 2], [0, 3]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        gold_pose_stack_block_in_and_first_out, pose_stack_block_in_and_first_out.cpu()
    )


def test_get_kfo_indices_for_atoms(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import (
        get_kfo_indices_for_atoms,
        get_kfo_atom_parents,
        get_children,
        get_id_and_frame_xyz,
    )

    torch_device = torch.device("cpu")
    device = torch_device

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=1, residue_end=3
    )

    res_not_connected = torch.zeros((1, 2, 2), dtype=torch.bool, device=torch_device)
    res_not_connected[0, 0, 0] = True  # simplest test case: not N-term
    res_not_connected[0, 1, 1] = True  # simplest test case: not C-term
    pose_stack = pose_stack_from_canonical_form(
        co, pbt, **canonical_form, res_not_connected=res_not_connected
    )
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    block_kfo_offset, kfo_2_orig_mapping, atom_kfo_index = get_kfo_indices_for_atoms(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pbt.n_atoms,
        pbt.atom_is_real,
    )
    fold_forest_parent = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    fold_forest_parent[0, 1] = 0

    ff_conn_to_parent = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    ff_conn_to_parent[0, 0] = 2  # jump
    ff_conn_to_parent[0, 1] = 0  # N->C

    block_in_out = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    block_in_out[0, 0, 0] = 3  # input from root
    block_in_out[0, 0, 1] = 1  # output through upper connection
    block_in_out[0, 1, 0] = 0  # input from lower connection
    block_in_out[0, 1, 1] = 1  # output through upper connection

    kfo_atom_parents, kfo_atom_grandparents = get_kfo_atom_parents(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        fold_forest_parent,
        # ff_conn_to_parent,
        block_in_out,
        pbt_gssps.parents,
        kfo_2_orig_mapping,
        atom_kfo_index,
        pbt_gssps.jump_atom,
        pbt.n_conn,
        pbt.conn_atom,
    )

    n_children, child_list_span, child_list, is_atom_jump = get_children(
        pose_stack.block_type_ind,
        block_in_out,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        pbt.n_conn,
    )

    id, frame_x, frame_y, frame_z, keep_atom_fixed = get_id_and_frame_xyz(
        pose_stack.coords.shape[1],
        pose_stack.block_coord_offset,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        child_list_span,
        child_list,
        is_atom_jump,
    )


def test_get_scans_for_two_copies_of_6_res_ubq_H(
    stack_of_two_six_res_ubqs, ff_2ubq_6res_H, torch_device
):

    pose_stack = stack_of_two_six_res_ubqs
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=torch_device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    raw_dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    assert raw_dofs is not None

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    kmd.forest.id,
                    kmd.forest.doftype,
                    kmd.forest.parent,
                    kmd.forest.frame_x,
                    kmd.forest.frame_y,
                    kmd.forest.frame_z,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        kmd.scan_data_fw.nodes,
        kmd.scan_data_fw.scans,
        kmd.scan_data_fw.gens,
        kmd.scan_data_bw.nodes,
        kmd.scan_data_bw.scans,
        kmd.scan_data_bw.gens,
        kinforest,
    )

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_get_scans_for_two_copies_of_6_res_ubq_U(
    stack_of_two_six_res_ubqs, ff_2ubq_6res_U, torch_device
):

    pose_stack = stack_of_two_six_res_ubqs
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_U)
    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)

    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=torch_device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    # get_c1_and_c2_atoms: jump atom 19, 18, 3
    # c1 c2 18 3
    # get_c1_and_c2_atoms: jump atom 74, 73, 59
    # c1 c2 73 59
    # get_c1_and_c2_atoms: jump atom 127, 126, 111
    # c1 c2 126 111
    # get_c1_and_c2_atoms: jump atom 182, 181, 167

    raw_dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    assert raw_dofs is not None

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    kmd.forest.id,
                    kmd.forest.doftype,
                    kmd.forest.parent,
                    kmd.forest.frame_x,
                    kmd.forest.frame_y,
                    kmd.forest.frame_z,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        kmd.scan_data_fw.nodes,
        kmd.scan_data_fw.scans,
        kmd.scan_data_fw.gens,
        kmd.scan_data_bw.nodes,
        kmd.scan_data_bw.scans,
        kmd.scan_data_bw.gens,
        kinforest,
    )

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_get_scans_for_two_copies_of_6_res_ubq_K(
    stack_of_two_six_res_ubqs, torch_device, ff_2ubq_6res_K
):
    pose_stack = stack_of_two_six_res_ubqs
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_K)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)

    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=torch_device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    # get_c1_and_c2_atoms: jump atom 19, 18, 3
    # c1 c2 18 3
    # get_c1_and_c2_atoms: jump atom 74, 73, 59
    # c1 c2 73 59
    # get_c1_and_c2_atoms: jump atom 127, 126, 111
    # c1 c2 126 111
    # get_c1_and_c2_atoms: jump atom 182, 181, 167

    raw_dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    assert raw_dofs is not None

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    kmd.forest.id,
                    kmd.forest.doftype,
                    kmd.forest.parent,
                    kmd.forest.frame_x,
                    kmd.forest.frame_y,
                    kmd.forest.frame_z,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        kmd.scan_data_fw.nodes,
        kmd.scan_data_fw.scans,
        kmd.scan_data_fw.gens,
        kmd.scan_data_bw.nodes,
        kmd.scan_data_bw.scans,
        kmd.scan_data_bw.gens,
        kinforest,
    )

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_kinmodule_construction_for_jagged_stack_H(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_H, torch_device
):

    pose_stack = jagged_stack_of_465_res_ubqs
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=torch_device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    raw_dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    assert raw_dofs is not None

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    kmd.forest.id,
                    kmd.forest.doftype,
                    kmd.forest.parent,
                    kmd.forest.frame_x,
                    kmd.forest.frame_y,
                    kmd.forest.frame_z,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        kmd.scan_data_fw.nodes,
        kmd.scan_data_fw.scans,
        kmd.scan_data_fw.gens,
        kmd.scan_data_bw.nodes,
        kmd.scan_data_bw.scans,
        kmd.scan_data_bw.gens,
        kinforest,
    )

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_kinmodule_construction_for_jagged_stack_star(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_star, torch_device
):

    pose_stack = jagged_stack_of_465_res_ubqs
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_star)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=torch_device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    raw_dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    assert raw_dofs is not None

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    kmd.forest.id,
                    kmd.forest.doftype,
                    kmd.forest.parent,
                    kmd.forest.frame_x,
                    kmd.forest.frame_y,
                    kmd.forest.frame_z,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        kmd.scan_data_fw.nodes,
        kmd.scan_data_fw.scans,
        kmd.scan_data_fw.gens,
        kmd.scan_data_bw.nodes,
        kmd.scan_data_bw.scans,
        kmd.scan_data_bw.gens,
        kinforest,
    )

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)
