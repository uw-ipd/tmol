_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "SimpleLJScore": ("test_lbfgs_armijo", "SimpleLJScore"),
    "test_lbfgs_armijo": ("test_lbfgs_armijo", "test_lbfgs_armijo"),
    "test_large_negative_gradient_does_not_converge": (
        "test_lbfgs_armijo",
        "test_large_negative_gradient_does_not_converge",
    ),
    "test_lbfgs_armijo_sparse": ("test_lbfgs_armijo", "test_lbfgs_armijo_sparse"),
    "test_lbfgs_armijo_short_history": (
        "test_lbfgs_armijo",
        "test_lbfgs_armijo_short_history",
    ),
    "test_batched_two_loop_matches_one_problem_at_a_time": (
        "test_lbfgs_segments",
        "test_batched_two_loop_matches_one_problem_at_a_time",
    ),
    "test_two_loop_ignores_zeroed_history_slots": (
        "test_lbfgs_segments",
        "test_two_loop_ignores_zeroed_history_slots",
    ),
    "BlockProblem": ("test_lbfgs_segments", "BlockProblem"),
    "test_segmented_min_matches_one_block_at_a_time": (
        "test_lbfgs_segments",
        "test_segmented_min_matches_one_block_at_a_time",
    ),
    "test_unsegmented_min_does_not_match": (
        "test_lbfgs_segments",
        "test_unsegmented_min_does_not_match",
    ),
    "test_constant_energy_offset_does_not_change_where_a_block_stops": (
        "test_lbfgs_segments",
        "test_constant_energy_offset_does_not_change_where_a_block_stops",
    ),
    "test_noise_floor_stops_before_grinding_on_noise": (
        "test_lbfgs_segments",
        "test_noise_floor_stops_before_grinding_on_noise",
    ),
    "test_only_stationary_segments_are_frozen": (
        "test_lbfgs_segments",
        "test_only_stationary_segments_are_frozen",
    ),
    "test_failed_line_search_retires_only_that_segment": (
        "test_lbfgs_segments",
        "test_failed_line_search_retires_only_that_segment",
    ),
    "test_gradtol_uses_gradient_magnitude": (
        "test_lbfgs_segments",
        "test_gradtol_uses_gradient_magnitude",
    ),
    "test_stack_size_does_not_change_a_block": (
        "test_lbfgs_segments",
        "test_stack_size_does_not_change_a_block",
    ),
    "test_mixed_stack_does_not_change_a_block": (
        "test_lbfgs_segments",
        "test_mixed_stack_does_not_change_a_block",
    ),
    "test_lbfgs_two_loop_matches_reference": (
        "test_lbfgs_two_loop",
        "test_lbfgs_two_loop_matches_reference",
    ),
    "test_lbfgs_two_loop_benchmark": (
        "test_lbfgs_two_loop",
        "test_lbfgs_two_loop_benchmark",
    ),
    "test_build_kinforest_sfxn_network_smoke": (
        "test_minimizers",
        "test_build_kinforest_sfxn_network_smoke",
    ),
    "test_run_kin_min_smoke": ("test_minimizers", "test_run_kin_min_smoke"),
    "test_run_cart_min_smoke": ("test_minimizers", "test_run_cart_min_smoke"),
    "test_run_kin_min_torch_lbfgs": ("test_minimizers", "test_run_kin_min_torch_lbfgs"),
    "test_score_stack_of_distinct_poses_matches_individual": (
        "test_pose_stack_minimization",
        "test_score_stack_of_distinct_poses_matches_individual",
    ),
    "test_cart_min_stack_of_distinct_poses": (
        "test_pose_stack_minimization",
        "test_cart_min_stack_of_distinct_poses",
    ),
    "test_kin_min_stack_of_distinct_poses": (
        "test_pose_stack_minimization",
        "test_kin_min_stack_of_distinct_poses",
    ),
    "test_cart_network_segment_ids": (
        "test_pose_stack_minimization",
        "test_cart_network_segment_ids",
    ),
    "test_kin_network_segment_ids": (
        "test_pose_stack_minimization",
        "test_kin_network_segment_ids",
    ),
    "test_cart_min_stack_of_identical_poses": (
        "test_pose_stack_minimization",
        "test_cart_min_stack_of_identical_poses",
    ),
    "test_cart_minimize_w_pose_and_sfxn_smoke": (
        "test_scorefunction_minimization",
        "test_cart_minimize_w_pose_and_sfxn_smoke",
    ),
    "test_kin_minimize_w_pose_and_sfxn_smoke": (
        "test_scorefunction_minimization",
        "test_kin_minimize_w_pose_and_sfxn_smoke",
    ),
    "test_minimize_w_pose_and_sfxn_benchmark": (
        "test_scorefunction_minimization",
        "test_minimize_w_pose_and_sfxn_benchmark",
    ),
    "test_minimizer": ("test_scorefunction_minimization", "test_minimizer"),
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
