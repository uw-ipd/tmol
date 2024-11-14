import torch
import numpy
import attrs

from collections import defaultdict
from numba import jit

import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from tmol.types.torch import Tensor

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.kinematics.datatypes import (
    NodeType,
    # KinForest,
    # KinForestScanData,
    # KinematicModuleData,
)
from tmol.kinematics.dof_modules import KinematicModule2
from tmol.kinematics.fold_forest import EdgeType
from tmol.kinematics.scan_ordering import (
    construct_kin_module_data_for_pose,
    _annotate_block_type_with_gen_scan_path_segs,
    _annotate_packed_block_type_with_gen_scan_path_segs,
)
from tmol.kinematics.compiled import inverse_kin, forward_kin_op

from tmol.utility.tensor.common_operations import exclusive_cumsum1d

# @jit
# def get_branch_depth(parents):
#     # modeled off get_children
#     nelts = parents.shape[0]

#     n_immediate_children = numpy.full(nelts, 0, dtype=numpy.int32)
#     for i in range(nelts):
#         p = parents[i]
#         assert p <= i
#         if p == i:
#             continue
#         n_immediate_children[p] += 1

#     child_list = numpy.full(nelts, -1, dtype=numpy.int32)
#     child_list_span = numpy.empty((nelts, 2), dtype=numpy.int32)

#     child_list_span[0, 0] = 0
#     child_list_span[0, 1] = n_immediate_children[0]
#     for i in range(1, nelts):
#         child_list_span[i, 0] = child_list_span[i - 1, 1]
#         child_list_span[i, 1] = child_list_span[i, 0] + n_immediate_children[i]

#     # Pass 3, fill the child list for each parent.
#     # As we do this,


def test_gen_seg_scan_paths_block_type_annotation_smoke(fresh_default_restype_set):
    torch_device = torch.device("cpu")

    bt_list = [bt for bt in fresh_default_restype_set.residue_types if bt.name == "LEU"]
    for bt in bt_list:
        _annotate_block_type_with_gen_scan_path_segs(bt)
        assert hasattr(bt, "gen_seg_scan_path_segs")


def test_calculate_ff_edge_delays_for_two_res_ubq(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

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

    max_n_edges = 1
    ff_edges = torch.zeros(
        (pose_stack.n_poses, max_n_edges, 4),
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 1] = 0
    ff_edges[0, 0, 2] = 1
    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    assert result is not None


def test_calculate_ff_edge_delays_for_6_res_ubq(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    torch_device = torch.device("cpu")
    device = torch_device

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

    max_n_edges = 5
    ff_edges = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 0] = 0
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = 0
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = 1
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4

    ff_edges[0, 3, 0] = 0
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = 0
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
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
    # print("dfs_order_of_ff_edges", dfs_order_of_ff_edges)
    # print("n_ff_edges", n_ff_edges)
    # print("ff_edge_parent", ff_edge_parent)
    # print("first_ff_edge_for_block_cpu", first_ff_edge_for_block_cpu)
    # print("pose_stack_ff_parent", pose_stack_ff_parent)
    # print("max_gen_depth_of_ff_edge", max_gen_depth_of_ff_edge)
    # print("first_child_of_ff_edge", first_child_of_ff_edge)
    # print("delay_for_edge", delay_for_edge)
    # print("toposort_index_for_edge", toposort_index_for_edge)


def test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_H(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    torch_device = torch.device("cpu")
    device = torch_device

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
    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    max_n_edges = 5
    ff_edges = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 0] = 0
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = 0
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = 1
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4

    ff_edges[0, 3, 0] = 0
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = 0
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    # Let's flip the jump and root the tree at res 4
    ff_edges[1, 0, 0] = 0
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = 0
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = 1
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1

    ff_edges[1, 3, 0] = 0
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = 0
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    # print("result", result)
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
    # print("dfs_order_of_ff_edges", dfs_order_of_ff_edges)
    # print("n_ff_edges", n_ff_edges)
    # print("ff_edge_parent", ff_edge_parent)
    # print("first_ff_edge_for_block_cpu", first_ff_edge_for_block_cpu)
    # print("pose_stack_ff_parent", pose_stack_ff_parent)
    # print("max_gen_depth_of_ff_edge", max_gen_depth_of_ff_edge)
    # print("first_child_of_ff_edge", first_child_of_ff_edge)
    # print("delay_for_edge", delay_for_edge)
    # print("toposort_index_for_edge", toposort_index_for_edge)

    gold_dfs_order_of_ff_edges = torch.tensor(
        [[2, 4, 3, 1, 0], [4, 3, 2, 1, 0]], dtype=torch.int32
    )
    gold_n_ff_edges = torch.tensor([5, 5], dtype=torch.int32)
    gold_ff_edge_parent = torch.tensor(
        [[2, 2, -1, 2, 2], [2, 2, -1, 2, 2]], dtype=torch.int32
    )
    gold_first_ff_edge_for_block_cpu = torch.tensor(
        [[0, 2, 1, 3, 2, 4], [0, 2, 1, 3, 2, 4]], dtype=torch.int32
    )
    gold_pose_stack_ff_parent = torch.tensor(
        [[1, -1, 1, 4, 1, 4], [1, 4, 1, 4, -1, 4]], dtype=torch.int32
    )
    gold_max_gen_depth_of_ff_edge = torch.tensor(
        [[4, 4, 5, 4, 4], [4, 4, 5, 4, 4]], dtype=torch.int32
    )
    gold_first_child_of_ff_edge = torch.tensor(
        [[-1, -1, 3, -1, -1], [-1, -1, 0, -1, -1]], dtype=torch.int32
    )
    gold_delay_for_edge = torch.tensor(
        [[1, 1, 0, 0, 1], [0, 1, 0, 1, 1]], dtype=torch.int32
    )
    gold_toposort_index_for_edge = torch.tensor(
        [4, 5, 0, 1, 8, 3, 9, 2, 6, 7], dtype=torch.int32
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


def test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_U(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import calculate_ff_edge_delays

    torch_device = torch.device("cpu")
    device = torch_device

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
    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    max_n_edges = 3
    ff_edges_cpu = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 2
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.jump
    ff_edges_cpu[0, 1, 1] = 2
    ff_edges_cpu[0, 1, 2] = 5

    ff_edges_cpu[0, 2, 0] = EdgeType.polymer
    ff_edges_cpu[0, 2, 1] = 5
    ff_edges_cpu[0, 2, 2] = 3

    # Let's flip the jump and root the tree at res 5
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 2
    ff_edges_cpu[1, 0, 2] = 0

    ff_edges_cpu[1, 1, 0] = EdgeType.jump
    ff_edges_cpu[1, 1, 1] = 5
    ff_edges_cpu[1, 1, 2] = 2

    ff_edges_cpu[1, 2, 0] = EdgeType.polymer
    ff_edges_cpu[1, 2, 1] = 5
    ff_edges_cpu[1, 2, 2] = 3

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges_cpu,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    # print("result", result)
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
    # print("dfs_order_of_ff_edges", dfs_order_of_ff_edges)
    # print("n_ff_edges", n_ff_edges)
    # print("ff_edge_parent", ff_edge_parent)
    # print("first_ff_edge_for_block_cpu", first_ff_edge_for_block_cpu)
    # print("pose_stack_ff_parent", pose_stack_ff_parent)
    # print("max_gen_depth_of_ff_edge", max_gen_depth_of_ff_edge)
    # print("first_child_of_ff_edge", first_child_of_ff_edge)
    # print("delay_for_edge", delay_for_edge)
    # print("toposort_index_for_edge", toposort_index_for_edge)

    gold_dfs_order_of_ff_edges = torch.tensor([[1, 2, 0], [2, 1, 0]], dtype=torch.int32)
    gold_n_ff_edges = torch.tensor([3, 3], dtype=torch.int32)
    gold_ff_edge_parent = torch.tensor([[-1, 0, 1], [1, -1, 1]], dtype=torch.int32)
    gold_first_ff_edge_for_block_cpu = torch.tensor(
        [[0, 0, 0, 2, 2, 1], [0, 0, 1, 2, 2, 1]], dtype=torch.int32
    )
    gold_pose_stack_ff_parent = torch.tensor(
        [[1, 2, -1, 4, 5, 2], [1, 2, 5, 4, 5, -1]], dtype=torch.int32
    )
    gold_max_gen_depth_of_ff_edge = torch.tensor(
        [[4, 4, 4], [4, 4, 4]], dtype=torch.int32
    )
    gold_first_child_of_ff_edge = torch.tensor(
        [[-1, 2, -1], [-1, 0, -1]], dtype=torch.int32
    )
    gold_delay_for_edge = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.int32)
    gold_toposort_index_for_edge = torch.tensor([0, 3, 4, 2, 1, 5], dtype=torch.int32)

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


def test_calculate_parent_block_conn_in_and_out_for_two_copies_of_6_res_ubq(ubq_pdb):
    from tmol.kinematics.compiled.compiled_ops import (
        calculate_ff_edge_delays,
        get_block_parent_connectivity_from_toposort,
    )

    torch_device = torch.device("cpu")
    device = torch_device

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
    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    max_n_edges = 5
    ff_edges = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges[0, 0, 0] = 0
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = 0
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = 1
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4

    ff_edges[0, 3, 0] = 0
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = 0
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    # Let's flip the jump and root the tree at res 4
    ff_edges[1, 0, 0] = 0
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = 0
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = 1
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1

    ff_edges[1, 3, 0] = 0
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = 0
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    # print("result", result)
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
    ) = result
    pose_stack_block_in_and_first_out = get_block_parent_connectivity_from_toposort(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edges,
        first_ff_edge_for_block,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
        pbt.n_conn,
        pbt.polymeric_conn_inds,
    )
    # print("pose_stack_block_in_and_first_out", pose_stack_block_in_and_first_out)


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

    bt0 = pbt.active_block_types[pose_stack.block_type_ind[0, 0]]
    bt1 = pbt.active_block_types[pose_stack.block_type_ind[0, 1]]
    # print("bt0", bt0.name, bt0.n_atoms)
    # print("bt1", bt1.name, bt1.n_atoms)
    # print("n block types", pbt.n_types)

    block_kfo_offset, kfo_2_orig_mapping, atom_kfo_index = get_kfo_indices_for_atoms(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pbt.n_atoms,
        pbt.atom_is_real,
    )
    # print("block_kfo_offset", block_kfo_offset)
    # print("kfo_2_orig_mapping", kfo_2_orig_mapping)
    # print("atom_kfo_index", atom_kfo_index)

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

    # print("pose_stack.block_type_ind", pose_stack.block_type_ind.dtype)
    # print(
    #     "pose_stack.inter_residue_connections",
    #     pose_stack.inter_residue_connections.dtype,
    # )
    # print("fold_forest_parent", fold_forest_parent.dtype)
    # print("ff_conn_to_parent", ff_conn_to_parent.dtype)
    # print("block_in_out", block_in_out.dtype)
    # print("pbt_gssps.parents", pbt_gssps.parents.dtype)
    # print("kfo_2_orig_mapping", kfo_2_orig_mapping.dtype)
    # print("atom_kfo_index", atom_kfo_index.dtype)
    # print("pbt_gssps.jump_atom", pbt_gssps.jump_atom.dtype)
    # print("pbt.n_conn", pbt.n_conn.dtype)
    # print("pbt.conn_atom", pbt.conn_atom.dtype)

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

    # print("kfo_atom_parents", kfo_atom_parents)
    # print("kfo_atom_grandparents", kfo_atom_grandparents)

    n_children, child_list_span, child_list, is_atom_jump = get_children(
        pose_stack.block_type_ind,
        block_in_out,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        pbt.n_conn,
    )
    # print("n_children", n_children)
    # print("child_list_span", child_list_span)
    # print("child_list", child_list)
    # print("is_atom_jump", is_atom_jump)

    id, frame_x, frame_y, frame_z = get_id_and_frame_xyz(
        pose_stack.coords.shape[1],
        pose_stack.block_coord_offset,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        child_list_span,
        child_list,
        is_atom_jump,
    )
    # print("id", id)
    # print("frame_x", frame_x)
    # print("frame_y", frame_y)
    # print("frame_z", frame_z)


# other topologies we need to test:
# multiple jumps from single block
# "u" instead of "H" shaped FT:
# >1 residue in peptide edges of H shaped FT


def test_get_scans_for_two_copies_of_6_res_ubq_H(ubq_pdb):

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
    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    # pbt_gssps = pbt.gen_seg_scan_path_segs

    # print("pbt_gssps.scan_path_seg_is_inter_block")
    # print(pbt_gssps.scan_path_seg_is_inter_block[24, 0, 1])

    max_n_edges = 5
    ff_edges_cpu = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges_cpu[0, 0, 0] = 0
    ff_edges_cpu[0, 0, 1] = 1
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = 0
    ff_edges_cpu[0, 1, 1] = 1
    ff_edges_cpu[0, 1, 2] = 2

    ff_edges_cpu[0, 2, 0] = 1
    ff_edges_cpu[0, 2, 1] = 1
    ff_edges_cpu[0, 2, 2] = 4

    ff_edges_cpu[0, 3, 0] = 0
    ff_edges_cpu[0, 3, 1] = 4
    ff_edges_cpu[0, 3, 2] = 3

    ff_edges_cpu[0, 4, 0] = 0
    ff_edges_cpu[0, 4, 1] = 4
    ff_edges_cpu[0, 4, 2] = 5

    # Let's flip the jump and root the tree at res 4
    ff_edges_cpu[1, 0, 0] = 0
    ff_edges_cpu[1, 0, 1] = 1
    ff_edges_cpu[1, 0, 2] = 0

    ff_edges_cpu[1, 1, 0] = 0
    ff_edges_cpu[1, 1, 1] = 1
    ff_edges_cpu[1, 1, 2] = 2

    ff_edges_cpu[1, 2, 0] = 1
    ff_edges_cpu[1, 2, 1] = 4
    ff_edges_cpu[1, 2, 2] = 1

    ff_edges_cpu[1, 3, 0] = 0
    ff_edges_cpu[1, 3, 1] = 4
    ff_edges_cpu[1, 3, 2] = 3

    ff_edges_cpu[1, 4, 0] = 0
    ff_edges_cpu[1, 4, 1] = 4
    ff_edges_cpu[1, 4, 2] = 5

    # ff_edges_device = ff_edges_cpu.to(torch_device)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)

    # print("nodes_fw", kmd.scan_data_fw.nodes)
    # print("scans_fw", kmd.scan_data_fw.scans)
    # print("gens_fw", kmd.scan_data_fw.gens)
    # print("nodes_bw", kmd.scan_data_bw.nodes)
    # print("scans_bw", kmd.scan_data_bw.scans)
    # print("gens_bw", kmd.scan_data_bw.gens)

    kincoords = torch.zeros((kmd.forest.id.shape[0], 3), dtype=torch.float32)
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    # print("dof_type", dof_type)

    # get_c1_and_c2_atoms: jump atom 19, 18, 3
    # c1 c2 18 3
    # get_c1_and_c2_atoms: jump atom 74, 73, 59
    # c1 c2 73 59
    # get_c1_and_c2_atoms: jump atom 127, 126, 111
    # c1 c2 126 111
    # get_c1_and_c2_atoms: jump atom 182, 181, 167

    # def print_frames(jump, i):
    #     print(
    #         f"jump {jump}: dof_type[{i}] {dof_type[i]} frame_x[{i}] {frame_x[i]}, frame_y[{i}] {frame_y[i]}, frame_z[{i}] {frame_z[i]}"
    #     )

    # def print_children(jump, i):
    #     for child_ind in range(child_list_span[i], child_list_span[i + 1]):
    #         child = child_list[child_ind]
    #         print_frames(f"child of {jump}", child)

    # def print_three_frames(jump, at1, at2, at3):
    #     print_frames(jump, at1)
    #     print_children(jump, at1)
    #     print_frames(jump, at2)
    #     print_frames(jump, at3)

    # print_three_frames(1, 19, 18, 3)
    # print_three_frames(2, 74, 73, 59)
    # print_three_frames(3, 127, 126, 111)
    # print_three_frames(4, 182, 181, 167)

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

    # print("kincoords[35:45]", kincoords[35:45])
    # print("new_coords[35:45]", new_coords[35:45])

    # print("kincoords[0:10]", kincoords[0:10])
    # print("new_coords[0:10]", new_coords[0:10])

    # print("kincoords[20:30]", kincoords[20:30])
    # print("new_coords[20:30]", new_coords[20:30])

    # print("kincoords[100:110]", kincoords[100:110])
    # print("new_coords[100:110]", new_coords[100:110])

    # print("kincoords[120:130]", kincoords[120:130])
    # print("new_coords[120:130]", new_coords[120:130])

    # nz_diff = torch.nonzero(
    #     torch.logical_and(
    #         torch.abs(kincoords - new_coords) > 1e-5,
    #         torch.logical_not(torch.isnan(kincoords)),
    #     ),
    #     as_tuple=True,
    # )
    # print("diff", nz_diff[0][:10])
    # print("diff", nz_diff[1][:10])
    # print("kincoords", kincoords[nz_diff[:10]])
    # print("new_coords", new_coords[nz_diff[:10]])

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_get_scans_for_two_copies_of_6_res_ubq_U(ubq_pdb):

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
    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], torch_device)
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    # pbt_gssps = pbt.gen_seg_scan_path_segs

    # print("pbt_gssps.scan_path_seg_is_inter_block")
    # print(pbt_gssps.scan_path_seg_is_inter_block[24, 0, 1])

    max_n_edges = 3
    ff_edges_cpu = torch.full(
        (pose_stack.n_poses, max_n_edges, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 2
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.jump
    ff_edges_cpu[0, 1, 1] = 2
    ff_edges_cpu[0, 1, 2] = 5

    ff_edges_cpu[0, 2, 0] = EdgeType.polymer
    ff_edges_cpu[0, 2, 1] = 5
    ff_edges_cpu[0, 2, 2] = 3

    # Let's flip the jump and root the tree at res 5
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 2
    ff_edges_cpu[1, 0, 2] = 0

    ff_edges_cpu[1, 1, 0] = EdgeType.jump
    ff_edges_cpu[1, 1, 1] = 5
    ff_edges_cpu[1, 1, 2] = 2

    ff_edges_cpu[1, 2, 0] = EdgeType.polymer
    ff_edges_cpu[1, 2, 1] = 5
    ff_edges_cpu[1, 2, 2] = 3

    # ff_edges_device = ff_edges_cpu.to(torch_device)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)

    print("nodes_fw", kmd.scan_data_fw.nodes)
    print("scans_fw", kmd.scan_data_fw.scans)
    print("gens_fw", kmd.scan_data_fw.gens)
    # print("nodes_bw", kmd.scan_data_bw.nodes)
    # print("scans_bw", kmd.scan_data_bw.scans)
    # print("gens_bw", kmd.scan_data_bw.gens)

    kincoords = torch.zeros((kmd.forest.id.shape[0], 3), dtype=torch.float32)
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

    # print("dof_type", dof_type)

    # get_c1_and_c2_atoms: jump atom 19, 18, 3
    # c1 c2 18 3
    # get_c1_and_c2_atoms: jump atom 74, 73, 59
    # c1 c2 73 59
    # get_c1_and_c2_atoms: jump atom 127, 126, 111
    # c1 c2 126 111
    # get_c1_and_c2_atoms: jump atom 182, 181, 167

    # def print_frames(jump, i):
    #     print(
    #         f"jump {jump}: dof_type[{i}] {dof_type[i]} frame_x[{i}] {frame_x[i]}, frame_y[{i}] {frame_y[i]}, frame_z[{i}] {frame_z[i]}"
    #     )

    # def print_children(jump, i):
    #     for child_ind in range(child_list_span[i], child_list_span[i + 1]):
    #         child = child_list[child_ind]
    #         print_frames(f"child of {jump}", child)

    # def print_three_frames(jump, at1, at2, at3):
    #     print_frames(jump, at1)
    #     print_children(jump, at1)
    #     print_frames(jump, at2)
    #     print_frames(jump, at3)

    # print_three_frames(1, 19, 18, 3)
    # print_three_frames(2, 74, 73, 59)
    # print_three_frames(3, 127, 126, 111)
    # print_three_frames(4, 182, 181, 167)

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

    # print("kincoords[35:45]", kincoords[35:45])
    # print("new_coords[35:45]", new_coords[35:45])

    # print("kincoords[0:10]", kincoords[0:10])
    # print("new_coords[0:10]", new_coords[0:10])

    # print("kincoords[20:30]", kincoords[20:30])
    # print("new_coords[20:30]", new_coords[20:30])

    # print("kincoords[100:110]", kincoords[100:110])
    # print("new_coords[100:110]", new_coords[100:110])

    # print("kincoords[120:130]", kincoords[120:130])
    # print("new_coords[120:130]", new_coords[120:130])

    # nz_diff = torch.nonzero(
    #     torch.logical_and(
    #         torch.abs(kincoords - new_coords) > 1e-5,
    #         torch.logical_not(torch.isnan(kincoords)),
    #     ),
    #     as_tuple=True,
    # )
    # print("diff", nz_diff[0][:10])
    # print("diff", nz_diff[1][:10])
    # print("kincoords", kincoords[nz_diff[:10]])
    # print("new_coords", new_coords[nz_diff[:10]])

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)


def test_decide_scan_paths_for_foldforest(ubq_pdb):
    torch_device = torch.device("cpu")

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=10
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)
