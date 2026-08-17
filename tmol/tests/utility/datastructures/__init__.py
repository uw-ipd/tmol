_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "in_place_heap": ("test_in_place_heap", "in_place_heap"),
    "reverse_insert10_heap_structure": (
        "test_in_place_heap",
        "reverse_insert10_heap_structure",
    ),
    "test_heap_construction_1": ("test_in_place_heap", "test_heap_construction_1"),
    "test_heap_construction_2": ("test_in_place_heap", "test_heap_construction_2"),
    "test_heap_clear_and_reconstruction": (
        "test_in_place_heap",
        "test_heap_clear_and_reconstruction",
    ),
    "test_heap_clear_and_reconstruction_smaller_subset": (
        "test_in_place_heap",
        "test_heap_clear_and_reconstruction_smaller_subset",
    ),
    "test_heap_with_gaps": ("test_in_place_heap", "test_heap_with_gaps"),
    "test_heap_with_gaps2": ("test_in_place_heap", "test_heap_with_gaps2"),
    "test_heap_pop": ("test_in_place_heap", "test_heap_pop"),
    "test_heap_pop2": ("test_in_place_heap", "test_heap_pop2"),
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
