import torch
import numpy
import pytest

from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import (
    KinematicModuleData,
    n_movable_jump_dof_types,
    n_movable_bond_dof_types,
)
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.scan_ordering import (
    construct_kin_module_data_for_pose,
)


@pytest.fixture
def mm_for_two_six_res_ubqs_no_term(stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    ff_edges_cpu = ff_2ubq_6res_H

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack_and_kmd(pose_stack, kmd)

    return mm


def test_movemap_construction_from_init(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    ff_edges_cpu = ff_2ubq_6res_H

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap(
        pose_stack.n_poses,
        pose_stack.max_n_blocks,
        pbt.max_n_torsions,
        pose_stack.max_n_block_atoms,
        pose_stack.device,
    )

    np = pose_stack.n_poses
    mxnb = pose_stack.max_n_blocks

    mxnt = pbt.max_n_torsions
    mxnapb = pose_stack.max_n_block_atoms

    assert mm.move_all_jumps == False
    assert mm.move_all_mc == False
    assert mm.move_all_sc == False
    assert mm.move_all_named_torsions == False

    assert mm.move_jumps.shape == (np, mxnb)
    assert mm.move_jumps_mask.shape == (np, mxnb)
    assert mm.move_mcs.shape == (np, mxnb)
    assert mm.move_mcs_mask.shape == (np, mxnb)
    assert mm.move_scs.shape == (np, mxnb)
    assert mm.move_scs_mask.shape == (np, mxnb)
    assert mm.move_named_torsions.shape == (np, mxnb)
    assert mm.move_named_torsions_mask.shape == (np, mxnb)

    # data members on a per-DOF basis
    assert mm.move_mc.shape == (np, mxnb, mxnt)
    assert mm.move_mc_mask.shape == (np, mxnb, mxnt)
    assert mm.move_sc.shape == (np, mxnb, mxnt)
    assert mm.move_sc_mask.shape == (np, mxnb, mxnt)
    assert mm.move_named_torsion.shape == (np, mxnb, mxnt)
    assert mm.move_named_torsion_mask.shape == (np, mxnb, mxnt)

    # data members on a per-atom basis
    assert mm.move_jump_dof.shape == (np, mxnb, n_movable_jump_dof_types)
    assert mm.move_jump_dof_mask.shape == (np, mxnb, n_movable_jump_dof_types)
    assert mm.move_atom_dof.shape == (np, mxnb, mxnapb, n_movable_bond_dof_types)
    assert mm.move_atom_dof_mask.shape == (np, mxnb, mxnapb, n_movable_bond_dof_types)


def test_movemap_construction_from_helper(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    ff_edges_cpu = ff_2ubq_6res_H

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack_and_kmd(pose_stack, kmd)

    np = pose_stack.n_poses
    mxnb = pose_stack.max_n_blocks

    mxnt = pbt.max_n_torsions
    mxnapb = pose_stack.max_n_block_atoms

    assert mm.move_all_jumps == False
    assert mm.move_all_mc == False
    assert mm.move_all_sc == False
    assert mm.move_all_named_torsions == False

    assert mm.move_jumps.shape == (np, mxnb)
    assert mm.move_jumps_mask.shape == (np, mxnb)
    assert mm.move_mcs.shape == (np, mxnb)
    assert mm.move_mcs_mask.shape == (np, mxnb)
    assert mm.move_scs.shape == (np, mxnb)
    assert mm.move_scs_mask.shape == (np, mxnb)
    assert mm.move_named_torsions.shape == (np, mxnb)
    assert mm.move_named_torsions_mask.shape == (np, mxnb)

    # data members on a per-torsion basis
    assert mm.move_mc.shape == (np, mxnb, mxnt)
    assert mm.move_mc_mask.shape == (np, mxnb, mxnt)
    assert mm.move_sc.shape == (np, mxnb, mxnt)
    assert mm.move_sc_mask.shape == (np, mxnb, mxnt)
    assert mm.move_named_torsion.shape == (np, mxnb, mxnt)
    assert mm.move_named_torsion_mask.shape == (np, mxnb, mxnt)

    # data members on a per-DOF basis
    assert mm.move_jump_dof.shape == (np, mxnb, n_movable_jump_dof_types)
    assert mm.move_jump_dof_mask.shape == (np, mxnb, n_movable_jump_dof_types)
    assert mm.move_atom_dof.shape == (np, mxnb, mxnapb, n_movable_bond_dof_types)
    assert mm.move_atom_dof_mask.shape == (np, mxnb, mxnapb, n_movable_bond_dof_types)


def move_all_setter_name_for_doftype(doftype):
    if doftype == "mc":
        return "set_move_all_mc_tors_for_blocks"
    elif doftype == "sc":
        return "set_move_all_sc_tors_for_blocks"
    elif doftype == "named_torsion":
        return "set_move_all_named_torsions_for_blocks"
    else:
        raise ValueError(f"doftype {doftype} not recognized")


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_integer(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] == False
    assert mask[1, 4] == False

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(1, 4)

    assert var[1, 4] == True
    assert mask[1, 4] == True


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_boolean_mask(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] == False
    assert mask[1, 4] == False

    bool_mask = torch.zeros((2, 6), dtype=torch.bool)
    bool_mask[1, 4] = True

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(bool_mask)

    for i in range(2):
        for j in range(6):
            assert var[i, j] == bool_mask[i, j]
            assert mask[i, j] == bool_mask[i, j]


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_boolean_mask2(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] == False
    assert mask[1, 4] == False

    bool_mask = torch.zeros((2, 6), dtype=torch.bool)
    bool_mask[0, 1:3] = True
    bool_mask[1, 2:4] = True

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(bool_mask)

    for i in range(2):
        for j in range(6):
            assert var[i, j] == bool_mask[i, j]
            assert mask[i, j] == bool_mask[i, j]


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_boolean_masks(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] == False
    assert mask[1, 4] == False

    pose_mask = torch.zeros((2,), dtype=torch.bool)
    pose_mask[1] = True
    block_mask = torch.zeros((6,), dtype=torch.bool)
    block_mask[1:4] = True

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(pose_mask, block_mask)

    for i in range(2):
        for j in range(6):
            assert var[i, j] == (pose_mask[i] and block_mask[j])
            assert mask[i, j] == (pose_mask[i] and block_mask[j])


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_index_tensors(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] == False
    assert mask[1, 4] == False

    pose_index_tensor = torch.zeros((8,), dtype=torch.int64)
    pose_index_tensor[0:4] = 0
    pose_index_tensor[4:8] = 1
    block_index_tensor = torch.zeros((8,), dtype=torch.int64)
    block_index_tensor[0:4] = torch.arange(4)
    block_index_tensor[4:8] = torch.arange(4) + 1

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(pose_index_tensor, block_index_tensor)

    gold_standard_var = torch.zeros((2, 6), dtype=torch.bool)
    for i in range(8):
        gold_standard_var[pose_index_tensor[i], block_index_tensor[i]] = True

    for i in range(2):
        for j in range(6):
            assert var[i, j] == gold_standard_var[i, j]
            assert mask[i, j] == gold_standard_var[i, j]


def test_set_move_all_jump_dofs_for_jump_by_index(mm_for_two_six_res_ubqs_no_term):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_all_jump_dofs_for_jump(1, 0)
    assert mm.move_jumps[0, 0] == False
    assert mm.move_jumps_mask[0, 0] == False
    assert mm.move_jumps[1, 0] == True
    assert mm.move_jumps_mask[1, 0] == True


###################


def move_particular_setter_name_for_doftype(doftype):
    if doftype == "mc":
        return "set_move_mc_tor_for_blocks"
    elif doftype == "sc":
        return "set_move_sc_tor_for_blocks"
    elif doftype == "named_torsion":
        return "set_move_named_torsion_for_blocks"
    else:
        raise ValueError(f"doftype {doftype} not recognized")


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_particular_doftypes_for_block_by_integer(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)
    print("var", var.shape)

    assert var[1, 4, 1] == False
    assert mask[1, 4, 1] == False

    print(
        "move_particular_setter_name_for_doftype(doftype)",
        move_particular_setter_name_for_doftype(doftype),
    )
    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(1, 4, 1)

    print("var", var)

    assert var[1, 4, 1] == True
    assert mask[1, 4, 1] == True


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_particular_doftypes_for_block_by_boolean_mask(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    bool_mask = torch.zeros((2, 6, 7), dtype=torch.bool)
    bool_mask[1, 4, 3] = True

    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(bool_mask)

    for i in range(2):
        for j in range(6):
            for k in range(4):
                assert var[i, j, k] == bool_mask[i, j, k]
                assert mask[i, j, k] == bool_mask[i, j, k]


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_particular_doftypes_for_block_by_boolean_mask2(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    bool_mask = torch.zeros((2, 6, 7), dtype=torch.bool)
    bool_mask[0, 1:3, 4] = True
    bool_mask[1, 2:4, 2] = True

    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(bool_mask)

    for i in range(2):
        for j in range(6):
            for k in range(4):
                assert var[i, j, k] == bool_mask[i, j, k]
                assert mask[i, j, k] == bool_mask[i, j, k]


# @pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
# def test_set_move_particular_doftypes_for_block_by_boolean_masks(
#     doftype, mm_for_two_six_res_ubqs_no_term
# ):
#     mm = mm_for_two_six_res_ubqs_no_term

#     varname = f"move_{doftype}"
#     maskname = f"move_{doftype}_mask"

#     var = getattr(mm, varname)
#     mask = getattr(mm, maskname)

#     pose_mask = torch.zeros((2,), dtype=torch.bool)
#     pose_mask[1] = True
#     block_mask = torch.zeros((6,), dtype=torch.bool)
#     block_mask[1:4] = True
#     dof_mask = torch.zeros((7,), dtype=torch.bool)
#     dof_mask[1:3] = True

#     setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
#     setter(pose_mask, block_mask, dof_mask)

#     for i in range(2):
#         for j in range(6):
#             for k in range(4):
#                 assert var[i, j, k] == (pose_mask[i] and block_mask[j] and dof_mask[k])
#                 assert mask[i, j, k] == (pose_mask[i] and block_mask[j] and dof_mask[k])


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_particular_doftypes_for_block_by_index_tensors(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    pose_index_tensor = torch.zeros((8,), dtype=torch.int64)
    pose_index_tensor[0:4] = 0
    pose_index_tensor[4:8] = 1
    block_index_tensor = torch.zeros((8,), dtype=torch.int64)
    block_index_tensor[0:4] = torch.arange(4)
    block_index_tensor[4:8] = torch.arange(4) + 1
    dof_index_tensor = torch.zeros((8,), dtype=torch.int64)
    dof_index_tensor[0:4] = 3
    dof_index_tensor[4:8] = 2

    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(pose_index_tensor, block_index_tensor, dof_index_tensor)

    gold_standard_var = torch.zeros((2, 6, 7), dtype=torch.bool)
    for i in range(8):
        gold_standard_var[
            pose_index_tensor[i], block_index_tensor[i], dof_index_tensor[i]
        ] = True

    for i in range(2):
        for j in range(6):
            for k in range(4):
                assert var[i, j, k] == gold_standard_var[i, j, k]
                assert mask[i, j, k] == gold_standard_var[i, j, k]


def test_set_move_particular_jump_dofs_for_jump_by_index(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_jump_dof_for_jumps(1, 0, 0)
    assert mm.move_jump_dof[0, 0, 0] == False
    assert mm.move_jump_dof_mask[0, 0, 0] == False
    assert mm.move_jump_dof[1, 0, 0] == True
    assert mm.move_jump_dof_mask[1, 0, 0] == True


def test_set_move_particular_atom_dofs(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_atom_dof_for_blocks(1, 0, 2, 3)
    assert mm.move_atom_dof[1, 0, 2, 3] == True
    assert mm.move_atom_dof_mask[1, 0, 2, 3] == True


def test_set_move_particular_atom_dofs(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    full_index = torch.zeros([2, 6, mm.max_n_atoms_per_block, 4], dtype=torch.bool)
    full_index[1, 0, 2, 3] = True
    mm.set_move_atom_dof_for_blocks(full_index)
    assert mm.move_atom_dof[1, 0, 2, 3] == True
    assert mm.move_atom_dof_mask[1, 0, 2, 3] == True


def test_minimizermap_construction_smoke(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    ff_edges_cpu = ff_2ubq_6res_H

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack_and_kmd(pose_stack, kmd)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None

    assert torch.sum(minmap.dof_mask) == 36
