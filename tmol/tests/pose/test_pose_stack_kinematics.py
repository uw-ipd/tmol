import torch
import numpy

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pose.pose_kinematics import (
    get_bonds_for_named_torsions,
    get_all_bonds,
    get_polymeric_bonds_in_fold_forest,
    construct_pose_stack_kinforest,
)
from tmol.kinematics.check_fold_forest import mark_polymeric_bonds_in_foldforest_edges
from tmol.kinematics.fold_forest import FoldForest, EdgeType


def test_get_bonds_for_named_torsions(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:4], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:6], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)
    pbt = pose_stack.packed_block_types

    middle_bond_ats = get_bonds_for_named_torsions(pose_stack)

    def resolve_atom(pose_ind, res_ind, uaid):
        bt_for_res = pose_stack.block_type_ind[pose_ind, res_ind]
        if uaid.atom is None:
            conn_ind = pbt.active_block_types[bt_for_res].connection_to_cidx[
                uaid.connection
            ]
            other_res_ind = pose_stack.inter_residue_connections[
                pose_ind, res_ind, conn_ind, 0
            ]
            if other_res_ind == -1:
                return pose_ind, -1, -1
            other_res_conn = pose_stack.inter_residue_connections[
                pose_ind, res_ind, conn_ind, 1
            ]
            other_res_block_type = pose_stack.block_type_ind[pose_ind, other_res_ind]

            other_res_atom = pose_stack.packed_block_types.atom_downstream_of_conn[
                other_res_block_type, other_res_conn, uaid.bond_sep_from_conn
            ]
            return pose_ind, other_res_ind, other_res_atom
        else:
            return (
                pose_ind,
                res_ind,
                pbt.active_block_types[bt_for_res].atom_to_idx[uaid.atom],
            )

    def atom_ind_to_global_index(pose_ind, res_ind, at_ind):
        if res_ind == -1 or at_ind == -1:
            return -1
        return (
            pose_ind * pose_stack.max_n_pose_atoms
            + pose_stack.block_coord_offset[pose_ind, res_ind]
            + at_ind
        )

    tor_at_inds = []
    for pose_ind in range(2):
        for i in range(pose_stack.max_n_blocks):
            ibt = pose_stack.block_type_ind[pose_ind, i]
            if ibt < 0:
                continue
            for tor in pbt.active_block_types[ibt].torsions:
                at1 = atom_ind_to_global_index(*resolve_atom(pose_ind, i, tor.b))
                at2 = atom_ind_to_global_index(*resolve_atom(pose_ind, i, tor.c))
                if at1 != -1 and at2 != -1:
                    tor_at_inds.append((at1.cpu(), at2.cpu()))
    middle_bond_ats_gold = numpy.array(tor_at_inds, dtype=numpy.int64)
    numpy.testing.assert_equal(middle_bond_ats_gold, middle_bond_ats.cpu().numpy())


def test_get_pose_stack_bonds(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:24], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[6:17], torch_device
    )
    p3 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[9:22], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2, p3], torch_device)
    pbt = pose_stack.packed_block_types

    bonds = get_all_bonds(pose_stack)

    bonds_gold = []
    for i in range(pose_stack.n_poses):
        for j in range(pose_stack.max_n_blocks):
            ijbt_ind = pose_stack.block_type_ind[i, j]
            if ijbt_ind < 0:
                continue
            i_j_bond_indices = pbt.active_block_types[ijbt_ind].bond_indices
            for k in range(i_j_bond_indices.shape[0]):

                def bond_atom(el):
                    return (
                        i * pose_stack.max_n_pose_atoms
                        + pose_stack.block_coord_offset[i, j]
                        + i_j_bond_indices[k, el]
                    )

                bonds_gold.append((bond_atom(0).cpu(), bond_atom(1).cpu()))
    for i in range(pose_stack.n_poses):
        for j in range(pose_stack.max_n_blocks):
            ijbt_ind = pose_stack.block_type_ind[i, j]
            if ijbt_ind < 0:
                continue
            ijbt = pbt.active_block_types[ijbt_ind]

            for k in range(ijbt.ordered_connection_atoms.shape[0]):
                other_res_ind = pose_stack.inter_residue_connections[i, j, k, 0]
                if other_res_ind == -1:
                    continue
                other_res_conn_ind = pose_stack.inter_residue_connections[i, j, k, 1]
                other_bt_ind = pose_stack.block_type_ind[i, other_res_ind]
                other_bt = pbt.active_block_types[other_bt_ind]
                # other_res = pose_stack.residues[i][other_res_ind]

                bonds_gold.append(
                    (
                        (
                            i * pose_stack.max_n_pose_atoms
                            + pose_stack.block_coord_offset[i, j]
                            + ijbt.ordered_connection_atoms[k]
                        ).cpu(),
                        (
                            i * pose_stack.max_n_pose_atoms
                            + pose_stack.block_coord_offset[i, other_res_ind]
                            + other_bt.ordered_connection_atoms[other_res_conn_ind]
                        ).cpu(),
                    )
                )
    bonds_gold = numpy.array(bonds_gold, dtype=numpy.int64)
    numpy.testing.assert_equal(bonds_gold, bonds.cpu().numpy())


def polymeric_bond_inds_for_pose_stack(pose_stack, polymeric_connections_for_kinforest):
    bond_inds_gold = []
    bt_cpu = pose_stack.block_type_ind.cpu()
    cpu_pbt = pose_stack.packed_block_types.cpu()
    for i in range(pose_stack.n_poses):
        for j in range(pose_stack.max_n_blocks):
            if pose_stack.block_type_ind[i, j] < 0:
                continue
            for k in range(pose_stack.max_n_blocks):
                if pose_stack.block_type_ind[i, k] < 0:
                    continue
                if j == k:
                    continue
                bt_j = bt_cpu[i, j]
                bt_k = bt_cpu[i, k]
                if polymeric_connections_for_kinforest[i, j, k]:
                    if j < k:
                        conn_j = cpu_pbt.up_conn_inds[bt_j]
                        conn_k = cpu_pbt.down_conn_inds[bt_k]
                    else:
                        conn_j = cpu_pbt.down_conn_inds[bt_j]
                        conn_k = cpu_pbt.up_conn_inds[bt_k]
                    at_j = cpu_pbt.conn_atom[bt_j, conn_j]
                    at_k = cpu_pbt.conn_atom[bt_k, conn_k]
                    global_at_j = (
                        i * pose_stack.max_n_pose_atoms
                        + pose_stack.block_coord_offset[i, j]
                        + at_j
                    )
                    global_at_k = (
                        i * pose_stack.max_n_pose_atoms
                        + pose_stack.block_coord_offset[i, k]
                        + at_k
                    )
                    bond_inds_gold.append((global_at_j, global_at_k))
    return numpy.array(bond_inds_gold, dtype=numpy.int64)


def test_get_polymeric_bonds_in_fold_forest(ubq_res, default_database):
    torch_device = torch.device("cpu")

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:4], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:6], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)

    edges = numpy.full((2, 1, 4), -1, dtype=int)
    edges[:, :, 0] = EdgeType.polymer
    edges[:, :, 1] = 0
    edges[:, 0, 2] = numpy.array([4, 6], dtype=int) - 1

    polymeric_connections_for_kinforest = mark_polymeric_bonds_in_foldforest_edges(
        2, 6, edges
    )

    bond_inds_gold = polymeric_bond_inds_for_pose_stack(
        pose_stack, polymeric_connections_for_kinforest
    )

    polymeric_bonds_in_kinforest = get_polymeric_bonds_in_fold_forest(
        pose_stack, polymeric_connections_for_kinforest
    )

    numpy.testing.assert_equal(
        bond_inds_gold, polymeric_bonds_in_kinforest.cpu().numpy()
    )


def test_get_polymeric_bonds_in_fold_forest_3(ubq_res, default_database):
    torch_device = torch.device("cpu")

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:4], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:6], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)

    edges = numpy.full((2, 1, 4), -1, dtype=int)
    edges[:, :, 0] = EdgeType.polymer
    edges[:, :, 2] = 0
    edges[:, 0, 1] = numpy.array([4, 6], dtype=int) - 1

    polymeric_connections_for_kinforest = mark_polymeric_bonds_in_foldforest_edges(
        2, 6, edges
    )

    bond_inds_gold = polymeric_bond_inds_for_pose_stack(
        pose_stack, polymeric_connections_for_kinforest
    )

    polymeric_bonds_in_kinforest = get_polymeric_bonds_in_fold_forest(
        pose_stack, polymeric_connections_for_kinforest
    )

    numpy.testing.assert_equal(
        bond_inds_gold, polymeric_bonds_in_kinforest.cpu().numpy()
    )


def test_get_polymeric_bonds_in_fold_forest_c_to_n(ubq_res, default_database):
    torch_device = torch.device("cpu")

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:8], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:11], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:5], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)

    edges = numpy.full((3, 2, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.polymer
    edges[:, 0, 1] = 0
    edges[0, 0, 2] = 7
    edges[1, 0, 1] = 5
    edges[1, 0, 2] = 0
    edges[1, 1, 0] = EdgeType.polymer
    edges[1, 1, 1] = 5
    edges[1, 1, 2] = 10
    edges[2, 0, 2] = 4

    polymeric_connections_for_kinforest = mark_polymeric_bonds_in_foldforest_edges(
        3, 11, edges
    )

    bond_inds_gold = polymeric_bond_inds_for_pose_stack(
        pose_stack, polymeric_connections_for_kinforest
    )

    polymeric_bonds_in_kinforest = get_polymeric_bonds_in_fold_forest(
        pose_stack, polymeric_connections_for_kinforest
    )

    numpy.testing.assert_equal(
        bond_inds_gold, polymeric_bonds_in_kinforest.cpu().numpy()
    )


def test_construct_pose_stack_kinforest(ubq_res, default_database):
    torch_device = torch.device("cpu")

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:8], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:11], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:5], torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)

    edges = numpy.full((3, 2, 4), -1, dtype=int)
    edges[:, 0, 0] = EdgeType.polymer
    edges[:, 0, 1] = 0
    edges[0, 0, 2] = 7
    edges[1, 0, 1] = 5
    edges[1, 0, 2] = 0
    edges[1, 1, 0] = EdgeType.polymer
    edges[1, 1, 1] = 5
    edges[1, 1, 2] = 10
    edges[2, 0, 2] = 4

    fold_forest = FoldForest.polymeric_forest(pose_stack.n_res_per_pose)

    kinforest = construct_pose_stack_kinforest(pose_stack, fold_forest)

    # TO DO: make sure kinforest is properly constructed
    assert kinforest is not None
