_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ext": ("test_device_operations", "ext"),
    "test_forall": ("test_device_operations", "test_forall"),
    "test_forall_large": ("test_device_operations", "test_forall_large"),
    "test_forall_stacks": ("test_device_operations", "test_forall_stacks"),
    "test_forall_stacks_large": ("test_device_operations", "test_forall_stacks_large"),
    "test_foreach_combination_triple": (
        "test_device_operations",
        "test_foreach_combination_triple",
    ),
    "test_foreach_combination_triple_large": (
        "test_device_operations",
        "test_foreach_combination_triple_large",
    ),
    "test_foreach_workgroup": ("test_device_operations", "test_foreach_workgroup"),
    "test_foreach_workgroup_large": (
        "test_device_operations",
        "test_foreach_workgroup_large",
    ),
    "test_scan_inclusive": ("test_device_operations", "test_scan_inclusive"),
    "test_scan_inclusive_large": (
        "test_device_operations",
        "test_scan_inclusive_large",
    ),
    "test_scan_exclusive": ("test_device_operations", "test_scan_exclusive"),
    "test_scan_exclusive_large": (
        "test_device_operations",
        "test_scan_exclusive_large",
    ),
    "test_scan_and_return_total_inclusive": (
        "test_device_operations",
        "test_scan_and_return_total_inclusive",
    ),
    "test_scan_and_return_total_inclusive_large": (
        "test_device_operations",
        "test_scan_and_return_total_inclusive_large",
    ),
    "test_scan_and_return_total_exclusive": (
        "test_device_operations",
        "test_scan_and_return_total_exclusive",
    ),
    "test_scan_and_return_total_exclusive_large": (
        "test_device_operations",
        "test_scan_and_return_total_exclusive_large",
    ),
    "test_reduce": ("test_device_operations", "test_reduce"),
    "test_reduce_large": ("test_device_operations", "test_reduce_large"),
    "test_load_balancing_search": (
        "test_device_operations",
        "test_load_balancing_search",
    ),
    "test_load_balancing_search_large": (
        "test_device_operations",
        "test_load_balancing_search_large",
    ),
    "test_segmented_scan_inclusive": (
        "test_device_operations",
        "test_segmented_scan_inclusive",
    ),
    "test_segmented_scan_inclusive_large": (
        "test_device_operations",
        "test_segmented_scan_inclusive_large",
    ),
    "test_segmented_scan_exclusive": (
        "test_device_operations",
        "test_segmented_scan_exclusive",
    ),
    "test_segmented_scan_exclusive_large": (
        "test_device_operations",
        "test_segmented_scan_exclusive_large",
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
