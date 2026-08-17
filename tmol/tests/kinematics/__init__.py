import pytest
import numpy
from tmol.kinematics import EdgeType

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_mark_polymeric_bonds_in_foldforest_edges_1": (
        "test_check_fold_forest",
        "test_mark_polymeric_bonds_in_foldforest_edges_1",
    ),
    "test_mark_polymeric_bonds_in_foldforest_edges_2": (
        "test_check_fold_forest",
        "test_mark_polymeric_bonds_in_foldforest_edges_2",
    ),
    "test_mark_polymeric_bonds_in_foldforest_edges_3": (
        "test_check_fold_forest",
        "test_mark_polymeric_bonds_in_foldforest_edges_3",
    ),
    "test_bfs_proper_forest_1": ("test_check_fold_forest", "test_bfs_proper_forest_1"),
    "test_bfs_proper_forest_2": ("test_check_fold_forest", "test_bfs_proper_forest_2"),
    "count_max_n_edges": ("test_check_fold_forest", "count_max_n_edges"),
    "test_validate_fold_forest_1": (
        "test_check_fold_forest",
        "test_validate_fold_forest_1",
    ),
    "test_validate_fold_forest_2": (
        "test_check_fold_forest",
        "test_validate_fold_forest_2",
    ),
    "test_validate_fold_forest_2b": (
        "test_check_fold_forest",
        "test_validate_fold_forest_2b",
    ),
    "test_validate_fold_forest_2c": (
        "test_check_fold_forest",
        "test_validate_fold_forest_2c",
    ),
    "test_validate_fold_forest_3": (
        "test_check_fold_forest",
        "test_validate_fold_forest_3",
    ),
    "test_validate_fold_forest_4": (
        "test_check_fold_forest",
        "test_validate_fold_forest_4",
    ),
    "test_validate_fold_forest_5": (
        "test_check_fold_forest",
        "test_validate_fold_forest_5",
    ),
    "test_validate_fold_forest_6": (
        "test_check_fold_forest",
        "test_validate_fold_forest_6",
    ),
    "test_validate_fold_forest_7": (
        "test_check_fold_forest",
        "test_validate_fold_forest_7",
    ),
    "test_validate_fold_forest_7b": (
        "test_check_fold_forest",
        "test_validate_fold_forest_7b",
    ),
    "test_gen_seg_scan_paths_block_type_annotation_smoke": (
        "test_create_scan_orering_from_block_types",
        "test_gen_seg_scan_paths_block_type_annotation_smoke",
    ),
    "test_calculate_ff_edge_delays_for_two_res_ubq": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_ff_edge_delays_for_two_res_ubq",
    ),
    "test_calculate_ff_edge_delays_for_6_res_ubq": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_ff_edge_delays_for_6_res_ubq",
    ),
    "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_H": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_H",
    ),
    "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_U": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_U",
    ),
    "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_K": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_ff_edge_delays_for_two_copies_of_6_res_ubq_K",
    ),
    "test_calculate_parent_block_conn_in_and_out_for_two_copies_of_6_res_ubq": (
        "test_create_scan_orering_from_block_types",
        "test_calculate_parent_block_conn_in_and_out_for_two_copies_of_6_res_ubq",
    ),
    "test_get_kfo_indices_for_atoms": (
        "test_create_scan_orering_from_block_types",
        "test_get_kfo_indices_for_atoms",
    ),
    "test_get_scans_for_two_copies_of_6_res_ubq_H": (
        "test_create_scan_orering_from_block_types",
        "test_get_scans_for_two_copies_of_6_res_ubq_H",
    ),
    "test_get_scans_for_two_copies_of_6_res_ubq_U": (
        "test_create_scan_orering_from_block_types",
        "test_get_scans_for_two_copies_of_6_res_ubq_U",
    ),
    "test_get_scans_for_two_copies_of_6_res_ubq_K": (
        "test_create_scan_orering_from_block_types",
        "test_get_scans_for_two_copies_of_6_res_ubq_K",
    ),
    "test_kinmodule_construction_for_jagged_stack_H": (
        "test_create_scan_orering_from_block_types",
        "test_kinmodule_construction_for_jagged_stack_H",
    ),
    "test_kinmodule_construction_for_jagged_stack_star": (
        "test_create_scan_orering_from_block_types",
        "test_kinmodule_construction_for_jagged_stack_star",
    ),
    "test_reasonable_fold_forest_smoke": (
        "test_fold_forest",
        "test_reasonable_fold_forest_smoke",
    ),
    "test_jagged_reasonable_fold_forest": (
        "test_fold_forest",
        "test_jagged_reasonable_fold_forest",
    ),
    "test_fold_forest_numbers_only_true_jumps": (
        "test_fold_forest",
        "test_fold_forest_numbers_only_true_jumps",
    ),
    "kinatom_to_atom_name": ("test_move_map", "kinatom_to_atom_name"),
    "mm_for_two_six_res_ubqs_no_term": (
        "test_move_map",
        "mm_for_two_six_res_ubqs_no_term",
    ),
    "mm_for_jagged_465_ubqs": ("test_move_map", "mm_for_jagged_465_ubqs"),
    "test_movemap_construction_from_init": (
        "test_move_map",
        "test_movemap_construction_from_init",
    ),
    "test_movemap_construction_from_helper": (
        "test_move_map",
        "test_movemap_construction_from_helper",
    ),
    "move_all_setter_name_for_doftype": (
        "test_move_map",
        "move_all_setter_name_for_doftype",
    ),
    "test_set_move_all_doftypes_for_block_by_integer": (
        "test_move_map",
        "test_set_move_all_doftypes_for_block_by_integer",
    ),
    "test_set_move_all_doftypes_for_block_by_boolean_mask": (
        "test_move_map",
        "test_set_move_all_doftypes_for_block_by_boolean_mask",
    ),
    "test_set_move_all_doftypes_for_block_by_boolean_mask2": (
        "test_move_map",
        "test_set_move_all_doftypes_for_block_by_boolean_mask2",
    ),
    "test_set_move_all_doftypes_for_block_by_boolean_masks": (
        "test_move_map",
        "test_set_move_all_doftypes_for_block_by_boolean_masks",
    ),
    "test_set_move_all_doftypes_for_block_by_index_tensors": (
        "test_move_map",
        "test_set_move_all_doftypes_for_block_by_index_tensors",
    ),
    "test_set_move_all_jump_dofs_for_jump_by_index": (
        "test_move_map",
        "test_set_move_all_jump_dofs_for_jump_by_index",
    ),
    "test_set_move_all_jump_dofs_for_root_jump_by_index": (
        "test_move_map",
        "test_set_move_all_jump_dofs_for_root_jump_by_index",
    ),
    "move_particular_setter_name_for_doftype": (
        "test_move_map",
        "move_particular_setter_name_for_doftype",
    ),
    "test_set_move_particular_doftypes_for_block_by_integer": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_integer",
    ),
    "test_set_move_particular_doftypes_for_block_by_integer_jagged": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_integer_jagged",
    ),
    "test_set_move_particular_doftypes_for_block_by_boolean_mask": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_boolean_mask",
    ),
    "test_set_move_particular_doftypes_for_block_by_boolean_mask_jagged": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_boolean_mask_jagged",
    ),
    "test_set_move_particular_doftypes_for_block_by_boolean_mask2": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_boolean_mask2",
    ),
    "test_set_move_particular_doftypes_for_block_by_index_tensors": (
        "test_move_map",
        "test_set_move_particular_doftypes_for_block_by_index_tensors",
    ),
    "test_set_move_particular_jump_dofs_for_jump_by_index": (
        "test_move_map",
        "test_set_move_particular_jump_dofs_for_jump_by_index",
    ),
    "test_set_move_particular_jump_dofs_for_root_jump_by_index": (
        "test_move_map",
        "test_set_move_particular_jump_dofs_for_root_jump_by_index",
    ),
    "test_set_move_particular_atom_dofs": (
        "test_move_map",
        "test_set_move_particular_atom_dofs",
    ),
    "test_set_move_particular_atom_dofs2": (
        "test_move_map",
        "test_set_move_particular_atom_dofs2",
    ),
    "enabled_phi_dof_atoms_from_minimizer_map": (
        "test_move_map",
        "enabled_phi_dof_atoms_from_minimizer_map",
    ),
    "test_minimizermap_construction_2_sixres_ubq_just_sc": (
        "test_move_map",
        "test_minimizermap_construction_2_sixres_ubq_just_sc",
    ),
    "test_minimizermap_construction_2_sixres_ubq_just_bb": (
        "test_move_map",
        "test_minimizermap_construction_2_sixres_ubq_just_bb",
    ),
    "test_minimizermap_construction_2_sixres_ubq": (
        "test_move_map",
        "test_minimizermap_construction_2_sixres_ubq",
    ),
    "test_minimizermap_construction_2_sixres_ubq_root_jump_min": (
        "test_move_map",
        "test_minimizermap_construction_2_sixres_ubq_root_jump_min",
    ),
    "test_minimizermap_construction_jagged_465_ubq": (
        "test_move_map",
        "test_minimizermap_construction_jagged_465_ubq",
    ),
    "test_minimizermap_construction_jagged_465_ubq_just_sc": (
        "test_move_map",
        "test_minimizermap_construction_jagged_465_ubq_just_sc",
    ),
    "test_minimizermap_construction_jagged_465_ubq_just_mc": (
        "test_move_map",
        "test_minimizermap_construction_jagged_465_ubq_just_mc",
    ),
    "test_minimizermap_construction_jagged_465_ubq_named_dofs": (
        "test_move_map",
        "test_minimizermap_construction_jagged_465_ubq_named_dofs",
    ),
    "kinforest_from_roots_and_bonds": (
        "test_scan_ordering",
        "kinforest_from_roots_and_bonds",
    ),
    "test_get_scans_simple_path": ("test_scan_ordering", "test_get_scans_simple_path"),
    "test_get_scans_two_simple_paths": (
        "test_scan_ordering",
        "test_get_scans_two_simple_paths",
    ),
    "test_get_scans_three_simple_paths": (
        "test_scan_ordering",
        "test_get_scans_three_simple_paths",
    ),
    "test_get_scans_three_simple_branches": (
        "test_scan_ordering",
        "test_get_scans_three_simple_branches",
    ),
    "kop_gradcheck_report": ("test_script_modules", "kop_gradcheck_report"),
    "kincoords_and_dofs_for_pose_stack_system": (
        "test_script_modules",
        "kincoords_and_dofs_for_pose_stack_system",
    ),
    "coord_weights_for_device": ("test_script_modules", "coord_weights_for_device"),
    "coord_weights": ("test_script_modules", "coord_weights"),
    "pose_stack_system1": ("test_script_modules", "pose_stack_system1"),
    "pose_stack_gradcheck_test_system1": (
        "test_script_modules",
        "pose_stack_gradcheck_test_system1",
    ),
    "pose_stack_system2": ("test_script_modules", "pose_stack_system2"),
    "pose_stack_gradcheck_test_system2": (
        "test_script_modules",
        "pose_stack_gradcheck_test_system2",
    ),
    "test_pose_stack_kinematics_module_smoke": (
        "test_script_modules",
        "test_pose_stack_kinematics_module_smoke",
    ),
    "test_pose_stack_kinematic_torch_op_gradcheck_perturbed": (
        "test_script_modules",
        "test_pose_stack_kinematic_torch_op_gradcheck_perturbed",
    ),
    "test_pose_stack_kinematic_torch_op_gradcheck_perturbed2": (
        "test_script_modules",
        "test_pose_stack_kinematic_torch_op_gradcheck_perturbed2",
    ),
    "test_pose_stack_kinematic_torch_op_gradcheck": (
        "test_script_modules",
        "test_pose_stack_kinematic_torch_op_gradcheck",
    ),
    "test_pose_stack_kinematics_op_device": (
        "test_script_modules",
        "test_pose_stack_kinematics_op_device",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@pytest.fixture
def ff_2ubq_6res_H():
    max_n_edges = 6
    ff_edges = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
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

    # Let's flip the jump and root the tree at res 4
    ff_edges[1, 0, 0] = EdgeType.polymer
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = EdgeType.polymer
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = EdgeType.jump
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1
    ff_edges[1, 2, 3] = 0

    ff_edges[1, 3, 0] = EdgeType.polymer
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = EdgeType.polymer
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    ff_edges[1, 5, 0] = EdgeType.root_jump
    ff_edges[1, 5, 1] = -1
    ff_edges[1, 5, 2] = 4

    return ff_edges


@pytest.fixture
def ff_3_jagged_ubq_465res_H():
    max_n_edges = 6
    ff_edges = numpy.full(
        (3, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    # 4 res pose
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = EdgeType.jump
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 3
    ff_edges[0, 2, 3] = 0

    ff_edges[0, 3, 0] = EdgeType.root_jump
    ff_edges[0, 3, 1] = -1
    ff_edges[0, 3, 2] = 1

    # 6 res pose
    ff_edges[1, 0, 0] = EdgeType.polymer
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = EdgeType.polymer
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = EdgeType.jump
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1
    ff_edges[1, 2, 3] = 0

    ff_edges[1, 3, 0] = EdgeType.polymer
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = EdgeType.polymer
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    ff_edges[1, 5, 0] = EdgeType.root_jump
    ff_edges[1, 5, 1] = -1
    ff_edges[1, 5, 2] = 4

    # 5 res Pose
    ff_edges[2, 0, 0] = EdgeType.polymer
    ff_edges[2, 0, 1] = 1
    ff_edges[2, 0, 2] = 0

    ff_edges[2, 1, 0] = EdgeType.polymer
    ff_edges[2, 1, 1] = 1
    ff_edges[2, 1, 2] = 2

    ff_edges[2, 2, 0] = EdgeType.jump
    ff_edges[2, 2, 1] = 4
    ff_edges[2, 2, 2] = 1
    ff_edges[2, 2, 3] = 0

    ff_edges[2, 3, 0] = EdgeType.polymer
    ff_edges[2, 3, 1] = 4
    ff_edges[2, 3, 2] = 3

    ff_edges[2, 4, 0] = EdgeType.root_jump
    ff_edges[2, 4, 1] = -1
    ff_edges[2, 4, 2] = 4

    return ff_edges


@pytest.fixture
def ff_3_jagged_ubq_465res_star():
    max_n_edges = 6
    ff_edges = numpy.full(
        (3, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    for i, nres in enumerate([4, 6, 5]):
        for j in range(nres):
            ff_edges[i, j, 0] = EdgeType.root_jump
            ff_edges[i, j, 1] = -1
            ff_edges[i, j, 2] = j

    return ff_edges


@pytest.fixture
def ff_2ubq_6res_U():
    max_n_edges = 4
    ff_edges_cpu = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 2
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.jump
    ff_edges_cpu[0, 1, 1] = 2
    ff_edges_cpu[0, 1, 2] = 5
    ff_edges_cpu[0, 1, 3] = 0

    ff_edges_cpu[0, 2, 0] = EdgeType.polymer
    ff_edges_cpu[0, 2, 1] = 5
    ff_edges_cpu[0, 2, 2] = 3

    ff_edges_cpu[0, 3, 0] = EdgeType.root_jump
    ff_edges_cpu[0, 3, 1] = -1
    ff_edges_cpu[0, 3, 2] = 2

    # Let's flip the jump and root the tree at res 5
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 2
    ff_edges_cpu[1, 0, 2] = 0

    ff_edges_cpu[1, 1, 0] = EdgeType.jump
    ff_edges_cpu[1, 1, 1] = 5
    ff_edges_cpu[1, 1, 2] = 2
    ff_edges_cpu[1, 1, 3] = 0

    ff_edges_cpu[1, 2, 0] = EdgeType.polymer
    ff_edges_cpu[1, 2, 1] = 5
    ff_edges_cpu[1, 2, 2] = 3

    ff_edges_cpu[1, 3, 0] = EdgeType.root_jump
    ff_edges_cpu[1, 3, 1] = -1
    ff_edges_cpu[1, 3, 2] = 5

    return ff_edges_cpu


@pytest.fixture
def ff_2ubq_6res_K():
    max_n_edges = 6
    ff_edges_cpu = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 1
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.polymer
    ff_edges_cpu[0, 1, 1] = 1
    ff_edges_cpu[0, 1, 2] = 2

    ff_edges_cpu[0, 2, 0] = EdgeType.jump
    ff_edges_cpu[0, 2, 1] = 1
    ff_edges_cpu[0, 2, 2] = 3
    ff_edges_cpu[0, 2, 3] = 0

    ff_edges_cpu[0, 3, 0] = EdgeType.jump
    ff_edges_cpu[0, 3, 1] = 1
    ff_edges_cpu[0, 3, 2] = 4
    ff_edges_cpu[0, 3, 3] = 1

    ff_edges_cpu[0, 4, 0] = EdgeType.polymer
    ff_edges_cpu[0, 4, 1] = 4
    ff_edges_cpu[0, 4, 2] = 5

    ff_edges_cpu[0, 5, 0] = EdgeType.root_jump
    ff_edges_cpu[0, 5, 1] = -1
    ff_edges_cpu[0, 5, 2] = 1

    # Let's flip everything
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 4
    ff_edges_cpu[1, 0, 2] = 3

    ff_edges_cpu[1, 1, 0] = EdgeType.polymer
    ff_edges_cpu[1, 1, 1] = 4
    ff_edges_cpu[1, 1, 2] = 5

    ff_edges_cpu[1, 2, 0] = EdgeType.jump
    ff_edges_cpu[1, 2, 1] = 4
    ff_edges_cpu[1, 2, 2] = 2
    ff_edges_cpu[1, 2, 3] = 0

    ff_edges_cpu[1, 3, 0] = EdgeType.jump
    ff_edges_cpu[1, 3, 1] = 4
    ff_edges_cpu[1, 3, 2] = 1
    ff_edges_cpu[1, 3, 3] = 1

    ff_edges_cpu[1, 4, 0] = EdgeType.polymer
    ff_edges_cpu[1, 4, 1] = 1
    ff_edges_cpu[1, 4, 2] = 0

    ff_edges_cpu[1, 5, 0] = EdgeType.root_jump
    ff_edges_cpu[1, 5, 1] = -1
    ff_edges_cpu[1, 5, 2] = 4

    return ff_edges_cpu
