_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "approx_for": ("test_cubic_hermite_polynomial", "approx_for"),
    "real": ("test_cubic_hermite_polynomial", "real"),
    "cubic_hermite_polynomial": (
        "test_cubic_hermite_polynomial",
        "cubic_hermite_polynomial",
    ),
    "test_unit_interpolate": ("test_cubic_hermite_polynomial", "test_unit_interpolate"),
    "test_unit_interpolate_to_zero": (
        "test_cubic_hermite_polynomial",
        "test_unit_interpolate_to_zero",
    ),
    "test_interpolate": ("test_cubic_hermite_polynomial", "test_interpolate"),
    "test_interpolate_to_zero": (
        "test_cubic_hermite_polynomial",
        "test_interpolate_to_zero",
    ),
    "get_notallclose_msg": ("test_energy_term", "get_notallclose_msg"),
    "assert_allclose": ("test_energy_term", "assert_allclose"),
    "print_table": ("test_energy_term", "print_table"),
    "pose_stack_from_pdb_and_resnums": (
        "test_energy_term",
        "pose_stack_from_pdb_and_resnums",
    ),
    "EnergyTermTestBase": ("test_energy_term", "EnergyTermTestBase"),
    "DummyEnergyTerm": ("test_energy_term", "DummyEnergyTerm"),
    "EnergyTermBaseTester": ("test_energy_term", "EnergyTermBaseTester"),
    "test_energy_term_base_write_baseline_smoke": (
        "test_energy_term",
        "test_energy_term_base_write_baseline_smoke",
    ),
    "test_energy_term_fail": ("test_energy_term", "test_energy_term_fail"),
    "test_condense_numpy_inds": ("test_stack_condense", "test_condense_numpy_inds"),
    "test_condense_torch_inds": ("test_stack_condense", "test_condense_torch_inds"),
    "test_take_values_w_sentineled_index1": (
        "test_stack_condense",
        "test_take_values_w_sentineled_index1",
    ),
    "test_take_values_w_sentineled_index_and_dest": (
        "test_stack_condense",
        "test_take_values_w_sentineled_index_and_dest",
    ),
    "test_condense_subset": ("test_stack_condense", "test_condense_subset"),
    "test_condense_numpy_inds_from_doc_string": (
        "test_stack_condense",
        "test_condense_numpy_inds_from_doc_string",
    ),
    "test_condense_torch_inds_from_doc_string": (
        "test_stack_condense",
        "test_condense_torch_inds_from_doc_string",
    ),
    "test_take_values_w_sentineled_index_from_doc_string": (
        "test_stack_condense",
        "test_take_values_w_sentineled_index_from_doc_string",
    ),
    "test_take_values_w_sentineled_index_and_dest_from_doc_string": (
        "test_stack_condense",
        "test_take_values_w_sentineled_index_and_dest_from_doc_string",
    ),
    "test_take_values_w_sentineled_dest_from_doc_string": (
        "test_stack_condense",
        "test_take_values_w_sentineled_dest_from_doc_string",
    ),
    "test_condense_subset_from_doc_string": (
        "test_stack_condense",
        "test_condense_subset_from_doc_string",
    ),
    "test_take_condensed_3d_subset_from_doc_string": (
        "test_stack_condense",
        "test_take_condensed_3d_subset_from_doc_string",
    ),
    "test_tile_subset_indices_torch": (
        "test_stack_condense",
        "test_tile_subset_indices_torch",
    ),
    "test_tile_subset_indices_torch2": (
        "test_stack_condense",
        "test_tile_subset_indices_torch2",
    ),
    "test_tile_subset_indices_numpy": (
        "test_stack_condense",
        "test_tile_subset_indices_numpy",
    ),
    "test_tile_subset_indices_numpy2": (
        "test_stack_condense",
        "test_tile_subset_indices_numpy2",
    ),
    "test_arg_tile_subset_indices_torch": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_torch",
    ),
    "test_arg_tile_subset_indices_torch2": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_torch2",
    ),
    "test_arg_tile_subset_indices_torch_w_max_n_entries": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_torch_w_max_n_entries",
    ),
    "test_arg_tile_subset_indices_numpy": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_numpy",
    ),
    "test_arg_tile_subset_indices_numpy2": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_numpy2",
    ),
    "test_arg_tile_subset_indices_numpy_w_max_n_entries": (
        "test_stack_condense",
        "test_arg_tile_subset_indices_numpy_w_max_n_entries",
    ),
    "uaid_pose_stack": ("test_uaid_util", "uaid_pose_stack"),
    "test_resolve_uaids_smoke": ("test_uaid_util", "test_resolve_uaids_smoke"),
    "test_resolve_uaids_intra_res": ("test_uaid_util", "test_resolve_uaids_intra_res"),
    "test_resolve_uaids_inter_res": ("test_uaid_util", "test_resolve_uaids_inter_res"),
    "test_resolve_uaids_inter_res2": (
        "test_uaid_util",
        "test_resolve_uaids_inter_res2",
    ),
    "test_resolve_uaids_unresolved_connection": (
        "test_uaid_util",
        "test_resolve_uaids_unresolved_connection",
    ),
    "test_resolve_unspecified_uaids": (
        "test_uaid_util",
        "test_resolve_unspecified_uaids",
    ),
    "warp_segreduce": ("test_warp_segreduce", "warp_segreduce"),
    "test_warp_segreduce_1": ("test_warp_segreduce", "test_warp_segreduce_1"),
    "test_warp_segreduce_vec3": ("test_warp_segreduce", "test_warp_segreduce_vec3"),
    "test_warp_segreduce_vec3_benchmark": (
        "test_warp_segreduce",
        "test_warp_segreduce_vec3_benchmark",
    ),
    "test_warp_segreduce_w_partial_warp": (
        "test_warp_segreduce",
        "test_warp_segreduce_w_partial_warp",
    ),
    "warp_stride_reduce": ("test_warp_stride_reduce", "warp_stride_reduce"),
    "test_warp_stride_reduce_full": (
        "test_warp_stride_reduce",
        "test_warp_stride_reduce_full",
    ),
    "test_warp_stride_reduce_full_vec3": (
        "test_warp_stride_reduce",
        "test_warp_stride_reduce_full_vec3",
    ),
    "test_warp_stride_reduce_w_partial_warp": (
        "test_warp_stride_reduce",
        "test_warp_stride_reduce_w_partial_warp",
    ),
    "resolve_uaids": ("uaid_util", "resolve_uaids"),
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
