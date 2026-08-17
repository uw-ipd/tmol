_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "TotalScoreParts": ("plot_score_component_pass", "TotalScoreParts"),
    "TotalScoreOnepass": ("plot_total_score_onepass", "TotalScoreOnepass"),
    "test_setup_block_type": ("test_atom_type_dependent_term", "test_setup_block_type"),
    "test_store_atom_types_in_packed_residue_types": (
        "test_atom_type_dependent_term",
        "test_store_atom_types_in_packed_residue_types",
    ),
    "test_take_heavyatom_inds_in_range": (
        "test_atom_type_dependent_term",
        "test_take_heavyatom_inds_in_range",
    ),
    "test_create_pose_bond_separation_two_ubq": (
        "test_bond_dependent_term",
        "test_create_pose_bond_separation_two_ubq",
    ),
    "test_pose_score_smoke": ("test_score_function", "test_pose_score_smoke"),
    "test_block_pair_scoring_matches_whole_pose": (
        "test_score_function",
        "test_block_pair_scoring_matches_whole_pose",
    ),
    "test_virtual_residue_scoring": (
        "test_score_function",
        "test_virtual_residue_scoring",
    ),
    "test_soft_score_function_all_score_types": (
        "test_score_function",
        "test_soft_score_function_all_score_types",
    ),
    "test_score_function_all_score_types": (
        "test_score_function",
        "test_score_function_all_score_types",
    ),
    "test_score_function_all_score_types_protein_dna": (
        "test_score_function",
        "test_score_function_all_score_types_protein_dna",
    ),
    "test_score_function_one_body_terms_getter": (
        "test_score_function",
        "test_score_function_one_body_terms_getter",
    ),
    "test_score_function_two_body_terms_getter": (
        "test_score_function",
        "test_score_function_two_body_terms_getter",
    ),
    "test_score_function_all_terms_getter": (
        "test_score_function",
        "test_score_function_all_terms_getter",
    ),
    "dont_test_res_centric_score_benchmark_setup": (
        "test_score_function_benchmarks",
        "dont_test_res_centric_score_benchmark_setup",
    ),
    "test_res_centric_score_benchmark": (
        "test_score_function_benchmarks",
        "test_res_centric_score_benchmark",
    ),
    "test_combined_res_centric_score_benchmark": (
        "test_score_function_benchmarks",
        "test_combined_res_centric_score_benchmark",
    ),
    "test_build_posestack": ("test_score_function_benchmarks", "test_build_posestack"),
    "test_render_module": ("test_score_function_benchmarks", "test_render_module"),
    "test_full": ("test_score_function_benchmarks", "test_full"),
    "test_build_coord_mask_and_minimize_for_first_residue": (
        "test_score_utils",
        "test_build_coord_mask_and_minimize_for_first_residue",
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
