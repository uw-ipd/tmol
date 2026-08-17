_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_stretch_i32": ("test_common_operations", "test_stretch_i32"),
    "test_stretch2_i32": ("test_common_operations", "test_stretch2_i32"),
    "test_exclusive_cumsum": ("test_common_operations", "test_exclusive_cumsum"),
    "test_nplus1d_tensor_from_list": (
        "test_common_operations",
        "test_nplus1d_tensor_from_list",
    ),
    "test_cat_diff_sized_tensors_w_same_sizes": (
        "test_common_operations",
        "test_cat_diff_sized_tensors_w_same_sizes",
    ),
    "test_cat_diff_sized_tensors_w_diff_sizes": (
        "test_common_operations",
        "test_cat_diff_sized_tensors_w_diff_sizes",
    ),
    "test_join_tensors_and_report_real_entries": (
        "test_common_operations",
        "test_join_tensors_and_report_real_entries",
    ),
    "test_invert_mapping": ("test_common_operations", "test_invert_mapping"),
    "tensor_accessor": ("test_tensor_accessor", "tensor_accessor"),
    "accessor_funcs": ("test_tensor_accessor", "accessor_funcs"),
    "test_tensor_vector_accessors": (
        "test_tensor_accessor",
        "test_tensor_vector_accessors",
    ),
    "matrix_accessor_funcs": ("test_tensor_accessor", "matrix_accessor_funcs"),
    "test_tensor_matrix_accessors": (
        "test_tensor_accessor",
        "test_tensor_matrix_accessors",
    ),
    "test_tensor_accessor_device_conversion": (
        "test_tensor_accessor",
        "test_tensor_accessor_device_conversion",
    ),
    "test_tensor_pack_eigen_matrix": (
        "test_tensor_accessor",
        "test_tensor_pack_eigen_matrix",
    ),
    "test_tensor_pack_constructors": (
        "test_tensor_accessor",
        "test_tensor_pack_constructors",
    ),
    "test_tview_slice": ("test_tensor_accessor", "test_tview_slice"),
    "tensor_collection": ("test_tensor_collection", "tensor_collection"),
    "test_tensor_collection": ("test_tensor_collection", "test_tensor_collection"),
    "tensor_struct": ("test_tensor_struct", "tensor_struct"),
    "test_tensor_struct": ("test_tensor_struct", "test_tensor_struct"),
    "test_tensor_view": ("test_tensor_struct", "test_tensor_view"),
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
