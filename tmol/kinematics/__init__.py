_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "mark_polymeric_bonds_in_foldforest_edges": (
        "check_fold_forest",
        "mark_polymeric_bonds_in_foldforest_edges",
    ),
    "bfs_proper_forest": ("check_fold_forest", "bfs_proper_forest"),
    "ensure_jumps_numbered_and_distinct": (
        "check_fold_forest",
        "ensure_jumps_numbered_and_distinct",
    ),
    "validate_fold_forest_jit": ("check_fold_forest", "validate_fold_forest_jit"),
    "validate_fold_forest": ("check_fold_forest", "validate_fold_forest"),
    "NodeType": ("datatypes", "NodeType"),
    "KinForest": ("datatypes", "KinForest"),
    "KinForestScanData": ("datatypes", "KinForestScanData"),
    "KinematicModuleData": ("datatypes", "KinematicModuleData"),
    "KinDOF": ("datatypes", "KinDOF"),
    "BondDOFTypes": ("datatypes", "BondDOFTypes"),
    "n_movable_bond_dof_types": ("datatypes", "n_movable_bond_dof_types"),
    "JumpDOFTypes": ("datatypes", "JumpDOFTypes"),
    "n_movable_jump_dof_types": ("datatypes", "n_movable_jump_dof_types"),
    "BondDOF": ("datatypes", "BondDOF"),
    "JumpDOF": ("datatypes", "JumpDOF"),
    "BTGenerationalSegScanPathSegs": ("datatypes", "BTGenerationalSegScanPathSegs"),
    "PBTGenerationalSegScanPathSegs": ("datatypes", "PBTGenerationalSegScanPathSegs"),
    "EdgeType": ("fold_forest", "EdgeType"),
    "FoldForest": ("fold_forest", "FoldForest"),
    "DOFTypes": ("metadata", "DOFTypes"),
    "DOFMetadata": ("metadata", "DOFMetadata"),
    "CartesianMoveMap": ("move_map", "CartesianMoveMap"),
    "MoveMap": ("move_map", "MoveMap"),
    "MinimizerMap": ("move_map", "MinimizerMap"),
    "CoordArray": ("operations", "CoordArray"),
    "inverseKin": ("operations", "inverseKin"),
    "get_children": ("scan_ordering", "get_children"),
    "get_scans": ("scan_ordering", "get_scans"),
    "KinForestScanOrdering": ("scan_ordering", "KinForestScanOrdering"),
    "construct_kin_module_data_for_pose": (
        "scan_ordering",
        "construct_kin_module_data_for_pose",
    ),
    "ResidueKinforestData": ("scan_ordering", "ResidueKinforestData"),
    "annotate_block_type_with_residue_kinforest_data": (
        "scan_ordering",
        "annotate_block_type_with_residue_kinforest_data",
    ),
    "PoseStackKinematicsModule": ("script_modules", "PoseStackKinematicsModule"),
    "_annotate_block_type_with_gen_scan_path_segs": (
        "scan_ordering",
        "_annotate_block_type_with_gen_scan_path_segs",
    ),
    "_annotate_packed_block_type_with_gen_scan_path_segs": (
        "scan_ordering",
        "_annotate_packed_block_type_with_gen_scan_path_segs",
    ),
    "_build_pose_fold_forest": ("fold_forest", "_build_pose_fold_forest"),
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
