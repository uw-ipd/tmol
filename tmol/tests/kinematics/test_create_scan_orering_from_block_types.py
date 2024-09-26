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
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.kinematics.datatypes import NodeType
from tmol.kinematics.fold_forest import EdgeType
from tmol.kinematics.scan_ordering import (
    # get_children,
    _annotate_block_type_with_gen_scan_paths,
    _annotate_packed_block_type_with_gen_scan_paths,
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
        _annotate_block_type_with_gen_scan_paths(bt)
        assert hasattr(bt, "gen_seg_scan_paths")


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
    _annotate_packed_block_type_with_gen_scan_paths(pbt)
    pbt_gssp = pbt.gen_seg_scan_paths

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
        pbt_gssp.scan_path_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssp.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssp.scan_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )


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
    _annotate_packed_block_type_with_gen_scan_paths(pbt)
    pbt_gssp = pbt.gen_seg_scan_paths

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
        pbt_gssp.scan_path_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssp.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssp.scan_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    # print("result", result)
    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        first_ff_edge_for_block_cpu,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        first_ff_edge_for_block,
        delay_for_edge,
    ) = result
    print("dfs_order_of_ff_edges", dfs_order_of_ff_edges)
    print("n_ff_edges", n_ff_edges)
    print("first_ff_edge_for_block_cpu", first_ff_edge_for_block_cpu)
    print("max_gen_depth_of_ff_edge", max_gen_depth_of_ff_edge)
    print("first_child_of_ff_edge", first_child_of_ff_edge)
    print("first_ff_edge_for_block", first_ff_edge_for_block)
    print("delay_for_edge", delay_for_edge)


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
    _annotate_packed_block_type_with_gen_scan_paths(pbt)
    pbt_gssp = pbt.gen_seg_scan_paths

    bt0 = pbt.active_block_types[pose_stack.block_type_ind[0, 0]]
    bt1 = pbt.active_block_types[pose_stack.block_type_ind[0, 1]]
    print("bt0", bt0.name, bt0.n_atoms)
    print("bt1", bt1.name, bt1.n_atoms)
    print("n block types", pbt.n_types)

    block_kfo_offset, kfo_2_orig_mapping, atom_kfo_index = get_kfo_indices_for_atoms(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pbt.n_atoms,
        pbt.atom_is_real,
    )
    print("block_kfo_offset", block_kfo_offset)
    print("kfo_2_orig_mapping", kfo_2_orig_mapping)
    print("atom_kfo_index", atom_kfo_index)

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

    print("pose_stack.block_type_ind", pose_stack.block_type_ind.dtype)
    print(
        "pose_stack.inter_residue_connections",
        pose_stack.inter_residue_connections.dtype,
    )
    print("fold_forest_parent", fold_forest_parent.dtype)
    print("ff_conn_to_parent", ff_conn_to_parent.dtype)
    print("block_in_out", block_in_out.dtype)
    print("pbt_gssp.parents", pbt_gssp.parents.dtype)
    print("kfo_2_orig_mapping", kfo_2_orig_mapping.dtype)
    print("atom_kfo_index", atom_kfo_index.dtype)
    print("pbt_gssp.jump_atom", pbt_gssp.jump_atom.dtype)
    print("pbt.n_conn", pbt.n_conn.dtype)
    print("pbt.conn_atom", pbt.conn_atom.dtype)

    kfo_atom_parents, kfo_atom_grandparents = get_kfo_atom_parents(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        fold_forest_parent,
        ff_conn_to_parent,
        block_in_out,
        pbt_gssp.parents,
        kfo_2_orig_mapping,
        atom_kfo_index,
        pbt_gssp.jump_atom,
        pbt.n_conn,
        pbt.conn_atom,
    )

    print("kfo_atom_parents", kfo_atom_parents)
    print("kfo_atom_grandparents", kfo_atom_grandparents)

    n_children, child_list_span, child_list, is_atom_jump = get_children(
        pose_stack.block_type_ind,
        ff_conn_to_parent,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        pbt.n_conn,
    )
    print("n_children", n_children)
    print("child_list_span", child_list_span)
    print("child_list", child_list)
    print("is_atom_jump", is_atom_jump)

    id, frame_x, frame_y, frame_z = get_id_and_frame_xyz(
        pose_stack.coords.shape[1],
        pose_stack.block_coord_offset,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        child_list_span,
        child_list,
        is_atom_jump,
    )
    print("id", id)
    print("frame_x", frame_x)
    print("frame_y", frame_y)
    print("frame_z", frame_z)


def test_construct_scan_paths_n_to_c_twores(ubq_pdb):
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
    _annotate_packed_block_type_with_gen_scan_paths(pbt)

    pbt_gssp = pbt.gen_seg_scan_paths

    # for bt in pbt.active_block_types:
    #     _annotate_block_type_with_gen_scan_paths(bt)

    # now lets assume we have everything we need for the final step
    # of kintree construction:

    # output will be:
    # (the data members of kintree)
    # id: Tensor[torch.int32][...]
    # # roots: Tensor[torch.int32][...] # not used in current kinforest
    # doftype: Tensor[torch.int32][...]
    # parent: Tensor[torch.int32][...]
    # frame_x: Tensor[torch.int32][...]
    # frame_y: Tensor[torch.int32][...]
    # frame_z: Tensor[torch.int32][...]
    # (and the data members appended in get_scans)
    # nodes
    # scans
    # gens

    # now we figure out: what data do we need to construct these things?

    bt0 = pbt.active_block_types[pose_stack.block_type_ind[0, 0]]
    bt1 = pbt.active_block_types[pose_stack.block_type_ind[0, 1]]
    print("bt0", bt0.name, bt0.n_atoms)
    print("bt1", bt1.name, bt1.n_atoms)
    bt0gssp = bt0.gen_seg_scan_paths
    bt1gssp = bt1.gen_seg_scan_paths

    print("nodes")
    print(bt0gssp.nodes_for_gen[3, 1])
    print(bt1gssp.nodes_for_gen[0, 1])

    print("scans")
    print(bt0gssp.scan_starts[3, 1])
    print(bt1gssp.scan_starts[0, 1])

    # print("gens")
    # print(bt0gssp.

    print("parents")
    print(bt0gssp.parents[3])
    print(bt1gssp.parents[0])
    print(
        "parents in pbt, res1",
        pbt_gssp.parents[pose_stack.block_type_ind[0, 0], 3],
    )
    print(
        "parents in pbt, res2",
        pbt_gssp.parents[pose_stack.block_type_ind[0, 1], 0],
    )

    ij0 = [3, 1]  # 3 => root "input"; Q: is this different from jump input?
    ij1 = [0, 1]

    nodes = numpy.zeros((bt0.n_atoms + bt1.n_atoms,), dtype=numpy.int32)
    scans = numpy.zeros(
        (max(bt0gssp.scan_starts.shape[2], bt1gssp.scan_starts.shape[2]),),
        dtype=numpy.int32,
    )
    # gens = numpy.zeros(())

    ids_gold = numpy.concatenate(
        (
            numpy.full((1,), -1, dtype=numpy.int32),
            numpy.arange(bt0.n_atoms + bt1.n_atoms, dtype=numpy.int32),
        )
    )
    print("ids_gold", ids_gold.shape)
    print("ids_gold", ids_gold)

    # fmt: off
    parents_gold = numpy.array(
        [
            0, # virtual root "atom"
            2, 0, 2, 3, 2, 5, 6, 7, 7, 1, 2, 5, 5, 6, 6, 9, 9, # res 1
            3, 18, 19, 20, 19, 22, 22, 23, 18, 19, 22, 23, 23, 24, 24, 24, 25, 25, 25,  # res 2
        ],
        dtype=numpy.int32,
    )
    # fmt: on
    print("parents_gold", parents_gold.shape)
    dof_type_gold = numpy.full(1 + bt0.n_atoms + bt1.n_atoms, 2, dtype=numpy.int32)
    dof_type_gold[0] = NodeType.root.value
    dof_type_gold[2] = NodeType.jump.value
    frame_x_gold = numpy.arange(1 + bt0.n_atoms + bt1.n_atoms, dtype=numpy.int32)
    frame_y_gold = parents_gold.copy()  # we will correct the jump atom below
    frame_z_gold = parents_gold[parents_gold]  # grandparents
    frame_x_gold[0] = 2
    frame_y_gold[0] = 0
    frame_z_gold[0] = 10
    frame_x_gold[2] = 2
    frame_y_gold[2] = 0
    frame_z_gold[2] = 10

    # fmt: off
    nodes_gold = numpy.array(
        [
            0, 2, 3, 18, 19, 20,  # gen 1
            2, 1, 2, 5, 6, 7, 9, 16, 2, 11, 3, 4, 18, 26, 19, 22, 23, 25, 34, 19, 27, 20, 21,  # gen 2
            5, 12, 5, 13, 1, 10, 6, 14, 6, 15, 7, 8, 9, 17, 22, 24, 31, 22, 28, 23, 29, 23, 30, 25, 35, 25, 36,  # gen 3
            24, 32, 24, 33,  # gen 4
        ],
        dtype=numpy.int32,
    )

    scans_gold = numpy.array(
        [
            0,  # gen 1
            0, 2, 8, 10, 12, 14, 19, 21,  # gen 2
            0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25,  # gen 3;
            0, 2,  # gen 4
        ],
        dtype=numpy.int32,
    )

    generations_gold = numpy.array(
        [
            [0, 0],
            [6, 1 + 0],
            [23 + 6, 8 + 1 + 0],
            [27 + 23 + 6, 13 + 8 + 1 + 0],
            [4 + 27 + 23 + 6, 2 + 13 + 8 + 1 + 0],
        ],
        dtype=numpy.int32,
    )
    # fmt: on

    print("nodes_gold", nodes_gold.shape)
    print("scans_gold", scans_gold.shape)
    print("generations_gold", generations_gold.shape)
    print("generations_gold", generations_gold)

    def _t(x):
        return torch.tensor(x, dtype=torch.int32)

    ids_gold_t = _t(ids_gold)
    parents_gold_t = _t(parents_gold)
    frame_x_gold_t = _t(frame_x_gold)
    frame_y_gold_t = _t(frame_y_gold)
    frame_z_gold_t = _t(frame_z_gold)
    dof_type_gold_t = _t(dof_type_gold)
    nodes_gold_t = _t(nodes_gold)
    scans_gold_t = _t(scans_gold)
    generations_gold_t = _t(generations_gold)

    kincoords = torch.zeros((1 + bt0.n_atoms + bt1.n_atoms, 3), dtype=torch.float32)
    kincoords[1:] = pose_stack.coords.view(-1, 3)[ids_gold[1:]]

    # okay, now what?
    # Let's test that the gold version of the kinforest will actually
    # generate the input coordinates given the dofs extracted from
    # the input coordinates
    raw_dofs = inverse_kin(
        kincoords,
        _t(parents_gold),
        _t(frame_x_gold),
        _t(frame_y_gold),
        _t(frame_z_gold),
        _t(dof_type_gold),
    )
    # print("raw dofs", raw_dofs.shape)
    # print("raw dofs", raw_dofs[:10])

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _tint(ts):
        return tuple(map(lambda t: t.to(torch.int32), ts))

    kinforest = _p(
        torch.stack(
            _tint(
                [
                    ids_gold_t,
                    dof_type_gold_t,
                    parents_gold_t,
                    frame_x_gold_t,
                    frame_y_gold_t,
                    frame_z_gold_t,
                ]
            ),
            dim=1,
        )
    )

    new_coords = forward_kin_op(
        raw_dofs,
        nodes_gold_t,
        scans_gold_t,
        generations_gold_t,
        nodes_gold_t,  # note: backward version; incorrect to assume same as forward, temp!
        scans_gold_t,
        generations_gold_t,
        kinforest,
    )

    # print("starting coords", pose_stack.coords.view(-1, 3)[14:19])

    # print("kincoords", kincoords[15:20])
    # print("new coords", new_coords[15:20])

    torch.testing.assert_close(kincoords, new_coords, rtol=1e-5, atol=1e-5)

    # okay: let's construct the components of the kinforest from
    # the block types

    # 1. id: Tensor[torch.int32][...]

    is_bt_real = pose_stack.block_type_ind != -1
    nz_is_bt_real = torch.nonzero(is_bt_real, as_tuple=True)
    bt_n_atoms = torch.zeros_like(pose_stack.block_type_ind64)
    bt_n_atoms[is_bt_real] = pbt.n_atoms[pose_stack.block_type_ind64[is_bt_real]].to(
        torch.int64
    )
    n_atoms_real_bt = bt_n_atoms[is_bt_real]
    n_nonroot_kin_atoms = bt_n_atoms.sum()
    n_kin_atoms = n_nonroot_kin_atoms + 1

    # let's imagine a variable that says for each residue
    # whether it is connected to its parent by a jump,
    # an N->C connection, or a C->N connection
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
        dtype=torch.int64,
        device=device,
    )
    block_in_out[0, 0, 0] = 3  # input from root
    block_in_out[0, 0, 1] = 1  # output through upper connection
    block_in_out[0, 1, 0] = 0  # input from lower connection
    block_in_out[0, 1, 1] = 1  # output through upper connection

    fold_forest_parent = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    fold_forest_parent[0, 1] = 0

    id = torch.concatenate(  # cat?
        (
            torch.full((1,), -1, dtype=torch.int32, device=device),
            torch.arange(n_nonroot_kin_atoms, dtype=torch.int32, device=device),
        )
    )
    torch.testing.assert_close(id, ids_gold_t)

    # doftype: Tensor[torch.int32][...]
    doftype = torch.full_like(id, NodeType.bond.value)

    # 2. parent: Tensor[torch.int32][...]

    parent = torch.full_like(id, -1, dtype=torch.int32, device=device)

    # masked-out residues and residues connected directly to the root
    # don't need their parent atoms calculated
    ffparent_is_real_block = fold_forest_parent != -1
    real_ffparent = fold_forest_parent[ffparent_is_real_block]
    nz_block_w_real_ffparent = torch.nonzero(ffparent_is_real_block, as_tuple=True)

    per_block_type_parent = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pbt.max_n_atoms),
        -1,
        dtype=torch.int32,
    )
    per_block_type_parent[is_bt_real, :] = pbt_gssp.parents[
        pose_stack.block_type_ind64[is_bt_real],
        block_in_out[is_bt_real][:, 0],
    ]
    print("per block type parent", per_block_type_parent)

    # atom_pose_ind = torch.arange(
    #     pose_stack.n_poses, dtype=torch.int32, device=device
    # ).unsqueeze(-1).unsqueeze(-1).expand(
    #     (pose_stack.n_poses, pose_stack.max_n_blocks, pose_stack.max_n_atoms)
    # )
    is_atom_real = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pose_stack.max_n_atoms),
        dtype=torch.bool,
    )
    is_atom_real[is_bt_real] = pbt.atom_is_real[pose_stack.block_type_ind64[is_bt_real]]

    # atom_block_coord_offset = pose_stack.block_coord_offset.unsqueeze(-1).expand(
    #     (pose_stack.n_poses, pose_stack.max_n_blocks, pose_stack.max_n_atoms)
    # )

    kfo_block_offset = bt_n_atoms.clone().flatten()
    kfo_block_offset[0] += 1  # add in the virtual root
    kfo_block_offset = exclusive_cumsum1d(kfo_block_offset)
    kfo_block_offset[0] = 1  # adjust for the virtual root
    kfo_block_offset = kfo_block_offset.view(
        (pose_stack.n_poses, pose_stack.max_n_blocks)
    )

    kfo_block_offset_for_atom = kfo_block_offset.unsqueeze(-1).expand(
        (pose_stack.n_poses, pose_stack.max_n_blocks, pose_stack.max_n_atoms)
    )
    real_bt_ind_for_bt = torch.full_like(
        pose_stack.block_type_ind, -1, dtype=torch.int32
    )
    real_bt_ind_for_bt[is_bt_real] = torch.arange(
        is_bt_real.to(torch.int32).sum(), dtype=torch.int32, device=device
    )

    # which atom on the parent are we connected to?
    # if we are connected by bond, then we can check the pose_stack's
    # inter_residue_connections tensor; if we are connected by jump,
    # then the parent atom is the jump atom of the parent block type
    real_ffparent_block_type = pose_stack.block_type_ind64[
        nz_block_w_real_ffparent[0], real_ffparent
    ]
    # not so fast, tiger
    # real_ffparent_conn_ind = pose_stack.inter_residue_connections[
    #     nz_block_w_real_ffparent[0], nz_block_w_real_ffparent[1], block_in_out[]
    # ]
    is_connected_to_ffparent_w_non_jump = torch.logical_and(
        ff_conn_to_parent != -1, ff_conn_to_parent != 2
    )
    nz_conn_to_ffparent_w_non_jump = torch.nonzero(
        is_connected_to_ffparent_w_non_jump, as_tuple=True
    )
    is_connected_to_root = ff_conn_to_parent == 2

    is_connected_to_ffparent_w_lower_conn = torch.logical_and(
        ff_conn_to_parent != -1, ff_conn_to_parent == 0
    )
    is_connected_to_ffparent_w_upper_conn = torch.logical_and(
        ff_conn_to_parent != -1, ff_conn_to_parent == 1
    )
    print(
        "is connected to ffparent w lower conn", is_connected_to_ffparent_w_lower_conn
    )
    print(
        "is connected to ffparent w upper conn", is_connected_to_ffparent_w_upper_conn
    )

    real_nonjump_ffparent = fold_forest_parent[is_connected_to_ffparent_w_non_jump]
    real_nonjump_ffparent_p_block_type = pose_stack.block_type_ind64[
        nz_conn_to_ffparent_w_non_jump[0], real_nonjump_ffparent
    ]
    real_nonjump_ffparent_block_type = pose_stack.block_type_ind64[
        nz_block_w_real_ffparent[0], nz_block_w_real_ffparent[1]
    ]

    conn_ind = torch.full_like(ff_conn_to_parent, -1, dtype=torch.int32)
    conn_ind[is_connected_to_ffparent_w_lower_conn] = pbt.down_conn_inds[
        pose_stack.block_type_ind64[is_connected_to_ffparent_w_lower_conn]
    ]
    conn_ind[is_connected_to_ffparent_w_upper_conn] = pbt.up_conn_inds[
        pose_stack.block_type_ind64[is_connected_to_ffparent_w_upper_conn]
    ]
    print("conn ind", conn_ind)
    real_nonjump_ffparent_p_conn_ind = pose_stack.inter_residue_connections[
        nz_conn_to_ffparent_w_non_jump[0],
        nz_conn_to_ffparent_w_non_jump[1],
        conn_ind[is_connected_to_ffparent_w_non_jump],
        1,
    ]
    real_nonjump_ffparent_p_conn_atom = (
        pbt.conn_atom[
            real_nonjump_ffparent_p_block_type, real_nonjump_ffparent_p_conn_ind
        ]
        + kfo_block_offset[nz_conn_to_ffparent_w_non_jump[0], real_nonjump_ffparent]
    )
    print("real_nonjump_ffparent_p_conn_atom", real_nonjump_ffparent_p_conn_atom)
    real_nonjump_ffparent_conn_atom = pbt.conn_atom[
        real_nonjump_ffparent_block_type, conn_ind[is_connected_to_ffparent_w_non_jump]
    ]
    atoms_connected_by_nonjump = (
        real_nonjump_ffparent_conn_atom
        + kfo_block_offset[
            nz_conn_to_ffparent_w_non_jump[0], nz_conn_to_ffparent_w_non_jump[1]
        ]
    )
    print("atoms connected by nonjump", atoms_connected_by_nonjump)

    # real_conn_to_root_conn_atom = pbt.conn_atom[
    #     pose_stack.block_type_ind64[is_connected_to_root], 0
    # ]
    real_conn_to_root_bt = pose_stack.block_type_ind64[is_connected_to_root]
    real_conn_to_root_atoms = pbt_gssp.jump_atom[real_conn_to_root_bt]
    atoms_connected_to_the_root = (
        real_conn_to_root_atoms + kfo_block_offset[is_connected_to_root]
    )

    # atoms_connected_to_the_root = 2  # TEMP! FIX ME!!!!
    print("atoms connected to the root")

    # TO DO:
    # Lookup jump conn atom when connected by jump

    parent[1:] = (
        per_block_type_parent[is_atom_real] + kfo_block_offset_for_atom[is_atom_real]
    )

    parent[atoms_connected_by_nonjump] = real_nonjump_ffparent_p_conn_atom.to(
        torch.int32
    )

    # correct the roots
    parent[0] = 0
    parent[atoms_connected_to_the_root] = 0

    # okay, but we have to adjust the parent atoms for the connection
    # atoms (with negative parent values)
    print("parent", parent)
    print("parents_gold_t", parents_gold_t)

    torch.testing.assert_close(parent, parents_gold_t)

    # # roots: Tensor[torch.int32][...] # not used in current kinforest

    # 3-5.
    # frame_x: Tensor[torch.int32][...]
    # frame_y: Tensor[torch.int32][...]
    # frame_z: Tensor[torch.int32][...]

    frame_x = torch.arange(n_kin_atoms, dtype=torch.int32, device=device)

    # 4-5:

    frame_y = parent
    grandparent = parent[parent]

    # needs correction!

    # Will fail currently w/o correction
    # torch.testing.assert_close(frame_x, frame_x_gold_t)

    # (and the data members appended in get_scans)
    # nodes
    # scans
    # gens


def test_decide_scan_paths_for_foldforest(ubq_pdb):
    torch_device = torch.device("cpu")

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=10
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)
