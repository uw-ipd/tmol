_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_build_missing_sidechains_jagged_pose_stack": (
        "test_build_missing_sidechains",
        "test_build_missing_sidechains_jagged_pose_stack",
    ),
    "test_build_missing_sidechains_no_optH": (
        "test_build_missing_sidechains",
        "test_build_missing_sidechains_no_optH",
    ),
    "ubq_ig": ("test_load_ig", "ubq_ig"),
    "test_load_ig": ("test_load_ig", "test_load_ig"),
    "construct_faux_rotamer_set_and_sparse_energies_table_from_ig": (
        "test_load_ig",
        "construct_faux_rotamer_set_and_sparse_energies_table_from_ig",
    ),
    "construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig": (
        "test_load_ig",
        "construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig",
    ),
    "test_construct_rotamer_set_and_sparse_energies_table_from_ig": (
        "test_load_ig",
        "test_construct_rotamer_set_and_sparse_energies_table_from_ig",
    ),
    "test_build_interaction_graph": ("test_load_ig", "test_build_interaction_graph"),
    "test_build_multi_pose_interaction_graph": (
        "test_load_ig",
        "test_build_multi_pose_interaction_graph",
    ),
    "test_run_single_pose_simA": ("test_load_ig", "test_run_single_pose_simA"),
    "test_run_two_poses_simA": ("test_load_ig", "test_run_two_poses_simA"),
    "setup_pose_stack_and_task": ("test_pack_rotamers", "setup_pose_stack_and_task"),
    "build_packer_energy_tables": ("test_pack_rotamers", "build_packer_energy_tables"),
    "run_pack_and_assert_scores": ("test_pack_rotamers", "run_pack_and_assert_scores"),
    "get_packer_sfxn": ("test_pack_rotamers", "get_packer_sfxn"),
    "get_constraints_only_sfxn": ("test_pack_rotamers", "get_constraints_only_sfxn"),
    "test_pack_rotamers": ("test_pack_rotamers", "test_pack_rotamers"),
    "test_pack_rotamers_optH": ("test_pack_rotamers", "test_pack_rotamers_optH"),
    "test_pack_rotamers_w_cst": ("test_pack_rotamers", "test_pack_rotamers_w_cst"),
    "test_pack_rotamers_w_empty_interaction_graph": (
        "test_pack_rotamers",
        "test_pack_rotamers_w_empty_interaction_graph",
    ),
    "test_pack_rotamers_w_dslf": ("test_pack_rotamers", "test_pack_rotamers_w_dslf"),
    "test_pack_rotamers2": ("test_pack_rotamers", "test_pack_rotamers2"),
    "test_pack_rotamers_irregular_sized_poses": (
        "test_pack_rotamers",
        "test_pack_rotamers_irregular_sized_poses",
    ),
    "test_packer_palette_smoke": ("test_packer_task", "test_packer_palette_smoke"),
    "test_packer_palette_design_to_canonical_aas": (
        "test_packer_task",
        "test_packer_palette_design_to_canonical_aas",
    ),
    "test_packer_palette_design_to_canonical_aas2_backward_compat": (
        "test_packer_task",
        "test_packer_palette_design_to_canonical_aas2_backward_compat",
    ),
    "test_packer_task_smoke": ("test_packer_task", "test_packer_task_smoke"),
    "test_residue_level_task_his_restrict_to_repacking_backward_compat": (
        "test_packer_task",
        "test_residue_level_task_his_restrict_to_repacking_backward_compat",
    ),
    "test_packer_task_ctor": ("test_packer_task", "test_packer_task_ctor"),
    "test_set_packer_task_ctor": ("test_packer_task", "test_set_packer_task_ctor"),
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
