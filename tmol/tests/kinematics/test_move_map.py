import torch

# import numpy
import pytest

from tmol.kinematics.move_map import MoveMap, MinimizerMap
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import (
    KinematicModuleData,
    n_movable_jump_dof_types,
    n_movable_bond_dof_types,
)

# from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.scan_ordering import (
    construct_kin_module_data_for_pose,
)


def kinatom_to_atom_name(
    pose_stack: PoseStack, kmd: KinematicModuleData, kin_atom: int
):
    pose_atom = kmd.forest.id[kin_atom]
    # print("kinatom_to_atom_name", kin_atom, "pose_atom", pose_atom)

    pose = pose_atom // pose_stack.max_n_pose_atoms
    pose_atom = pose_atom % pose_stack.max_n_pose_atoms
    # print("pose", pose, "pose_atom", pose_atom)

    nz_lt_offset = torch.nonzero(pose_stack.block_coord_offset[pose] > pose_atom)
    block = (
        nz_lt_offset[0].item() - 1
        if nz_lt_offset.shape[0] > 0
        else pose_stack.n_res_per_pose[pose] - 1
    )
    # print("block", block)
    # print("pose_stack.block_coord_offset[pose, block]", pose_stack.block_coord_offset[pose, block])

    block_type = pose_stack.block_type_ind[pose, block]
    block_name = pose_stack.packed_block_types.active_block_types[block_type].name
    # print("block_type", block_type, "block_name", block_name)
    atom_name = (
        pose_stack.packed_block_types.active_block_types[block_type]
        .atoms[pose_atom - pose_stack.block_coord_offset[pose, block]]
        .name
    )
    return pose, block, block_name, atom_name


@pytest.fixture
def mm_for_two_six_res_ubqs_no_term(stack_of_two_six_res_ubqs_no_term):
    return MoveMap.from_pose_stack(stack_of_two_six_res_ubqs_no_term)


@pytest.fixture
def mm_for_jagged_465_ubqs(jagged_stack_of_465_res_ubqs):
    return MoveMap.from_pose_stack(jagged_stack_of_465_res_ubqs)


def test_movemap_construction_from_init(stack_of_two_six_res_ubqs_no_term):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types

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

    assert mm.move_all_jumps is False
    assert mm.move_all_mc is False
    assert mm.move_all_sc is False
    assert mm.move_all_named_torsions is False

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


def test_movemap_construction_from_helper(stack_of_two_six_res_ubqs_no_term):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    pbt = pose_stack.packed_block_types
    mm = MoveMap.from_pose_stack(pose_stack)

    np = pose_stack.n_poses
    mxnb = pose_stack.max_n_blocks

    mxnt = pbt.max_n_torsions
    mxnapb = pose_stack.max_n_block_atoms

    assert mm.move_all_jumps is False
    assert mm.move_all_mc is False
    assert mm.move_all_sc is False
    assert mm.move_all_named_torsions is False

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

    assert var[1, 4] is False
    assert mask[1, 4] is False

    setter = getattr(mm, move_all_setter_name_for_doftype(doftype))
    setter(1, 4)

    assert var[1, 4]
    assert mask[1, 4]


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_all_doftypes_for_block_by_boolean_mask(
    doftype, mm_for_two_six_res_ubqs_no_term
):
    mm = mm_for_two_six_res_ubqs_no_term

    varname = f"move_{doftype}s"
    maskname = f"move_{doftype}s_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4] is False
    assert mask[1, 4] is False

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

    assert var[1, 4] is False
    assert mask[1, 4] is False

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

    assert var[1, 4] is False
    assert mask[1, 4] is False

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

    assert var[1, 4] is False
    assert mask[1, 4] is False

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
    assert mm.move_jumps[0, 0] is False
    assert mm.move_jumps_mask[0, 0] is False
    assert mm.move_jumps[1, 0]
    assert mm.move_jumps_mask[1, 0]


def test_set_move_all_jump_dofs_for_root_jump_by_index(mm_for_two_six_res_ubqs_no_term):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_all_jump_dofs_for_root_jump(0, 4)
    assert mm.move_root_jumps[0, 0] is False
    assert mm.move_root_jumps_mask[0, 0] is False
    assert mm.move_root_jumps[0, 4]
    assert mm.move_root_jumps_mask[0, 4]


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
    # print("var", var.shape)

    assert var[1, 4, 1] is False
    assert mask[1, 4, 1] is False

    # print(
    #     "move_particular_setter_name_for_doftype(doftype)",
    #     move_particular_setter_name_for_doftype(doftype),
    # )
    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(1, 4, 1)

    # print("var", var)

    assert var[1, 4, 1]
    assert mask[1, 4, 1]


@pytest.mark.parametrize("doftype", ["mc", "sc", "named_torsion"])
def test_set_move_particular_doftypes_for_block_by_integer_jagged(
    doftype, mm_for_jagged_465_ubqs
):
    mm = mm_for_jagged_465_ubqs

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    assert var[1, 4, 1] is False
    assert mask[1, 4, 1] is False

    setter = getattr(mm, move_particular_setter_name_for_doftype(doftype))
    setter(1, 4, 1)

    assert var[1, 4, 1]
    assert mask[1, 4, 1]


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
def test_set_move_particular_doftypes_for_block_by_boolean_mask_jagged(
    doftype, mm_for_jagged_465_ubqs
):
    mm = mm_for_jagged_465_ubqs

    varname = f"move_{doftype}"
    maskname = f"move_{doftype}_mask"

    var = getattr(mm, varname)
    mask = getattr(mm, maskname)

    bool_mask = torch.zeros((3, 6, 7), dtype=torch.bool)
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
    assert mm.move_jump_dof[0, 0, 0] is False
    assert mm.move_jump_dof_mask[0, 0, 0] is False
    assert mm.move_jump_dof[1, 0, 0]
    assert mm.move_jump_dof_mask[1, 0, 0]


def test_set_move_particular_jump_dofs_for_root_jump_by_index(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_jump_dof_for_root_jumps(1, 0, 0)
    assert mm.move_root_jump_dof[0, 0, 0] is False
    assert mm.move_root_jump_dof_mask[0, 0, 0] is False
    assert mm.move_root_jump_dof[1, 0, 0]
    assert mm.move_root_jump_dof_mask[1, 0, 0]


def test_set_move_particular_atom_dofs(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    mm.set_move_atom_dof_for_blocks(1, 0, 2, 3)
    assert mm.move_atom_dof[1, 0, 2, 3]
    assert mm.move_atom_dof_mask[1, 0, 2, 3]


def test_set_move_particular_atom_dofs2(
    mm_for_two_six_res_ubqs_no_term,
):
    mm = mm_for_two_six_res_ubqs_no_term
    full_index = torch.zeros([2, 6, mm.max_n_atoms_per_block, 4], dtype=torch.bool)
    full_index[1, 0, 2, 3] = True
    mm.set_move_atom_dof_for_blocks(full_index)
    assert mm.move_atom_dof[1, 0, 2, 3]
    assert mm.move_atom_dof_mask[1, 0, 2, 3]


def enabled_phi_dof_atoms_from_minimizer_map(pose_stack, kmd, minmap):
    n_poses = pose_stack.n_poses
    max_n_blocks = pose_stack.max_n_blocks
    enabled_dof_atoms = [
        [list() for _2 in range(max_n_blocks)] for _1 in range(n_poses)
    ]
    for i in range(minmap.dof_mask.shape[0]):
        if minmap.dof_mask[i, 3]:
            pose, block, _, atom_name = kinatom_to_atom_name(pose_stack, kmd, i)
            enabled_dof_atoms[pose][block].append(atom_name)
    return enabled_dof_atoms


def test_minimizermap_construction_2_sixres_ubq_just_sc(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    # pbt = pose_stack.packed_block_types
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_sc = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None

    # 14 chi, 3Q+2I+2F+1V+4K+2T
    assert torch.sum(minmap.dof_mask) == 28

    # enabled_dof_atoms = [[list() for _2 in range(6)] for _1 in range(2)]
    # for i in range(minmap.dof_mask.shape[0]):
    #     if minmap.dof_mask[i, 3]:
    #         pose, block, _, atom_name = kinatom_to_atom_name(pose_stack, kmd, i)
    #         enabled_dof_atoms[pose][block].append(atom_name)
    #         # print("Enabled DOF on atom i", i, "kinatom_to_atom_name(pose_stack, kmd, i)", kinatom_to_atom_name(pose_stack, kmd, i))
    enabled_dof_atoms = enabled_phi_dof_atoms_from_minimizer_map(
        pose_stack, kmd, minmap
    )

    dof_atoms_gold = [
        [
            ["CB", "CG", "CD"],
            ["CB", "CG1"],
            ["CB", "CG"],
            ["CB"],
            ["CB", "CG", "CD", "CE"],
            ["CB", "OG1"],
        ],
        [
            ["CB", "CG", "CD"],
            ["CB", "CG1"],
            ["CB", "CG"],
            ["CB"],
            ["CB", "CG", "CD", "CE"],
            ["CB", "OG1"],
        ],
    ]

    assert enabled_dof_atoms == dof_atoms_gold


def test_minimizermap_construction_2_sixres_ubq_just_bb(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    # print("ID:", kmd.forest.id[:10])
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_mc = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None

    # 3*6 backbone torsions, including the n- and c-term phi and psi dihedrals that extend
    # to the next (but absent) residues, since we have the no-termini versions here,
    # but not the omega dihedral on the c-term that would live on the N atom of i+1,
    # but also discounting the jump from the root to the first residue that disables
    # one backbone dihedral and the non-root jump also disables a second backbone
    # dihedral, but also discounting the backbone dihedral between residue 2 and 3
    # that is "destroyed" by the cutpoint (because residue 2 is N->C and residue 3
    # is C->N, and so the phi_c on residue 3's N atom looks like residue 2's omega
    # in the N->C direction and it looks like residue 3's phi in the C->N direction)
    # therefore 2*(3*6 - 1 - 2 - 1) == 28
    assert torch.sum(minmap.dof_mask) == 28

    enabled_dof_atoms = [[list() for _2 in range(6)] for _1 in range(2)]
    for i in range(minmap.dof_mask.shape[0]):
        if minmap.dof_mask[i, 3]:
            pose, block, _, atom_name = kinatom_to_atom_name(pose_stack, kmd, i)
            enabled_dof_atoms[pose][block].append(atom_name)
            # print("Enabled DOF on atom i", i, "kinatom_to_atom_name(pose_stack, kmd, i)", kinatom_to_atom_name(pose_stack, kmd, i))

    dof_atoms_gold = [
        [
            ["N", "CA", "C"],
            ["C"],
            ["N", "CA", "C"],
            ["N", "CA", "C"],
            ["C"],
            ["N", "CA", "C"],
        ],
        [
            ["N", "CA", "C"],
            ["C"],
            ["N", "CA", "C"],
            ["N", "CA", "C"],
            ["C"],
            ["N", "CA", "C"],
        ],
    ]

    assert enabled_dof_atoms == dof_atoms_gold


def test_minimizermap_construction_2_sixres_ubq(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    ff_edges_cpu = torch.tensor(ff_2ubq_6res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    gold_pose_stack_atom_for_jump = torch.tensor(
        [
            [[4, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            [[1, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ],
        dtype=torch.int32,
        device=pose_stack.device,
    )
    gold_pose_stack_atom_for_root_jump = torch.tensor(
        [[-1, 1, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
        dtype=torch.int32,
        device=pose_stack.device,
    )
    assert torch.all(kmd.pose_stack_atom_for_jump == gold_pose_stack_atom_for_jump)
    assert torch.all(
        kmd.pose_stack_atom_for_root_jump == gold_pose_stack_atom_for_root_jump
    )

    # print("pose_stack_atom_for_jump", kmd.pose_stack_atom_for_jump)
    # print("pose_stack_atom_for_root_jump", kmd.pose_stack_atom_for_root_jump)

    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None

    # 14 chi, 3Q+2I+2F+1V+4K+2T
    # 14 bb = 6*3 - 4
    # where the -4 is because two jumps will disable two phi_c dofs
    # 6 jump DOFs
    # enabled_dof_atoms = enabled_phi_dof_atoms_from_minimizer_map(pose_stack, kmd, minmap)
    # for i in range(2):
    #     for j in range(6):
    #         print(f"enabled_dof_atoms[{i}][{j}]", enabled_dof_atoms[i][j])

    pbt = pose_stack.packed_block_types
    jump_0_atom = (
        pose_stack.block_coord_offset[0, 4]
        + pbt.active_block_types[pose_stack.block_type_ind64[0, 4]].atom_to_idx["CA"]
    )
    root_jump_1_atom = (
        pose_stack.block_coord_offset[0, 1]
        + pbt.active_block_types[pose_stack.block_type_ind64[0, 1]].atom_to_idx["CA"]
    )
    jump_0_kinatom = torch.nonzero(kmd.forest.id == jump_0_atom)
    root_jump_1_kinatom = torch.nonzero(kmd.forest.id == root_jump_1_atom)
    # print("jump_0_kinatom", jump_0_kinatom)
    # print("root_jump_1_kinatom", root_jump_1_kinatom)
    # print("jump 0 dofs:", minmap.dof_mask[jump_0_kinatom, :], "sum", torch.sum(minmap.dof_mask[jump_0_kinatom, :6]))
    # print("root jump 1 dofs:", minmap.dof_mask[root_jump_1_kinatom, :], "sum", torch.sum(minmap.dof_mask[root_jump_1_kinatom, :6]))
    assert torch.sum(minmap.dof_mask[jump_0_kinatom, :6]) == 6
    assert torch.sum(minmap.dof_mask[root_jump_1_kinatom, :6]) == 0

    assert torch.sum(minmap.dof_mask) == 2 * (14 + 14 + 6)


def test_minimizermap_construction_2_sixres_ubq_root_jump_min(
    stack_of_two_six_res_ubqs_no_term, ff_2ubq_6res_H
):
    pose_stack = stack_of_two_six_res_ubqs_no_term
    # pbt = pose_stack.packed_block_types
    ff_edges_cpu = torch.from_numpy(ff_2ubq_6res_H).to(torch.int32)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_root_jumps = True
    mm.move_all_named_torsions = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None

    # 14 chi, 3Q+2I+2F+1V+4K+2T
    # 14 bb = 6*3 - 4
    # where the -4 is because two jumps will disable two phi_c dofs
    # 6 jump DOFs
    # 6 root jump DOFs
    assert torch.sum(minmap.dof_mask) == 2 * (14 + 14 + 6 + 6)

    pbt = pose_stack.packed_block_types
    jump_0_atom = (
        pose_stack.block_coord_offset[0, 4]
        + pbt.active_block_types[pose_stack.block_type_ind64[0, 4]].atom_to_idx["CA"]
    )
    root_jump_1_atom = (
        pose_stack.block_coord_offset[0, 1]
        + pbt.active_block_types[pose_stack.block_type_ind64[0, 1]].atom_to_idx["CA"]
    )
    jump_0_kinatom = torch.nonzero(kmd.forest.id == jump_0_atom)
    root_jump_1_kinatom = torch.nonzero(kmd.forest.id == root_jump_1_atom)
    # print("jump_0_kinatom", jump_0_kinatom)
    # print("root_jump_1_kinatom", root_jump_1_kinatom)
    # print("jump 0 dofs:", minmap.dof_mask[jump_0_kinatom, :], "sum", torch.sum(minmap.dof_mask[jump_0_kinatom, :6]))
    # print("root jump 1 dofs:", minmap.dof_mask[root_jump_1_kinatom, :], "sum", torch.sum(minmap.dof_mask[root_jump_1_kinatom, :6]))
    assert torch.sum(minmap.dof_mask[jump_0_kinatom, :6]) == 6
    assert torch.sum(minmap.dof_mask[root_jump_1_kinatom, :6]) == 6


def test_minimizermap_construction_jagged_465_ubq(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_H
):
    pose_stack = jagged_stack_of_465_res_ubqs
    # print("pose_stack.inter_residue_connections", pose_stack.inter_residue_connections)
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None
    # print("torch.sum(minmap.dof_mask)", torch.sum(minmap.dof_mask))
    assert torch.sum(minmap.dof_mask) == 27 + 36 + 6 * 3


def test_minimizermap_construction_jagged_465_ubq_just_sc(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_H
):
    pose_stack = jagged_stack_of_465_res_ubqs
    # print("pose_stack.inter_residue_connections", pose_stack.inter_residue_connections)
    # pbt = pose_stack.packed_block_types
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_sc = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None
    # print("torch.sum(minmap.dof_mask)", torch.sum(minmap.dof_mask))
    assert torch.sum(minmap.dof_mask) == 36

    enabled_dof_atoms = [[list() for _2 in range(6)] for _1 in range(3)]
    for i in range(minmap.dof_mask.shape[0]):
        if minmap.dof_mask[i, 3]:
            pose, block, _, atom_name = kinatom_to_atom_name(pose_stack, kmd, i)
            enabled_dof_atoms[pose][block].append(atom_name)
            # print("Enabled DOF on atom i", i, "kinatom_to_atom_name(pose_stack, kmd, i)", kinatom_to_atom_name(pose_stack, kmd, i))

    dof_atoms_gold = [
        [
            ["CB", "CG", "SD"],
            ["CB", "CG", "CD"],
            ["CB", "CG1"],
            ["CB", "CG"],
            [],
            [],
        ],
        [
            ["CB", "CG", "SD"],
            ["CB", "CG", "CD"],
            ["CB", "CG1"],
            ["CB", "CG"],
            ["CB"],
            ["CB", "CG", "CD", "CE"],
        ],
        [
            ["CB", "CG", "SD"],
            ["CB", "CG", "CD"],
            ["CB", "CG1"],
            ["CB", "CG"],
            ["CB"],
            [],
        ],
    ]

    assert enabled_dof_atoms == dof_atoms_gold


def test_minimizermap_construction_jagged_465_ubq_just_mc(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_H
):
    pose_stack = jagged_stack_of_465_res_ubqs
    # print("pose_stack.inter_residue_connections", pose_stack.inter_residue_connections)
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_mc = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None
    # print("torch.sum(minmap.dof_mask)", torch.sum(minmap.dof_mask))
    assert torch.sum(minmap.dof_mask) == 27

    enabled_dof_atoms = [[list() for _2 in range(6)] for _1 in range(3)]
    for i in range(minmap.dof_mask.shape[0]):
        if minmap.dof_mask[i, 3]:
            pose, block, _, atom_name = kinatom_to_atom_name(pose_stack, kmd, i)
            enabled_dof_atoms[pose][block].append(atom_name)
            # print("Enabled DOF on atom i", i, "kinatom_to_atom_name(pose_stack, kmd, i)", kinatom_to_atom_name(pose_stack, kmd, i))

    dof_atoms_gold = [
        [  # 6
            ["CA", "C"],
            ["C"],
            ["N", "CA", "C"],
            [],
            [],
            [],
        ],
        [  # 12
            ["CA", "C"],
            ["C"],
            ["N", "CA", "C"],
            ["N", "CA", "C"],
            ["C"],
            ["N", "CA"],
        ],
        [  # 9
            ["CA", "C"],
            ["C"],
            ["N", "CA", "C"],
            ["N", "CA", "C"],
            [],
            [],
        ],
    ]

    assert enabled_dof_atoms == dof_atoms_gold


def test_minimizermap_construction_jagged_465_ubq_named_dofs(
    jagged_stack_of_465_res_ubqs, ff_3_jagged_ubq_465res_H
):
    pose_stack = jagged_stack_of_465_res_ubqs
    # print("pose_stack.inter_residue_connections", pose_stack.inter_residue_connections)
    # pbt = pose_stack.packed_block_types
    ff_edges_cpu = torch.tensor(ff_3_jagged_ubq_465res_H)

    kmd = construct_kin_module_data_for_pose(pose_stack, ff_edges_cpu)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_named_torsions = True
    minmap = MinimizerMap(pose_stack, kmd, mm)
    assert minmap is not None
    # print("torch.sum(minmap.dof_mask)", torch.sum(minmap.dof_mask))
    assert torch.sum(minmap.dof_mask) == 27 + 36
