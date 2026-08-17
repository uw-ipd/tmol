_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "stretch": ("common_operations", "stretch"),
    "stretch2": ("common_operations", "stretch2"),
    "exclusive_cumsum1d": ("common_operations", "exclusive_cumsum1d"),
    "exclusive_cumsum2d": ("common_operations", "exclusive_cumsum2d"),
    "exclusive_cumsum2d_and_totals": (
        "common_operations",
        "exclusive_cumsum2d_and_totals",
    ),
    "print_row_numbered_tensor": ("common_operations", "print_row_numbered_tensor"),
    "nplus1d_tensor_from_list": ("common_operations", "nplus1d_tensor_from_list"),
    "cat_differently_sized_tensors": (
        "common_operations",
        "cat_differently_sized_tensors",
    ),
    "join_tensors_and_report_real_entries": (
        "common_operations",
        "join_tensors_and_report_real_entries",
    ),
    "invert_mapping": ("common_operations", "invert_mapping"),
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


"""Support utils for tensor data structures.

Includes c++ and python level utilities for tensor data manipulation.
"""
