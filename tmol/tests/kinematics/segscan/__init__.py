_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "extension": ("test_segscan", "extension"),
    "test_segscan_inclusive": ("test_segscan", "test_segscan_inclusive"),
    "test_segscan_exclusive": ("test_segscan", "test_segscan_exclusive"),
    "test_segscan": ("test_segscan", "test_segscan"),
    "test_segscan2": ("test_segscan", "test_segscan2"),
    "test_segscan_highly_segmented": ("test_segscan", "test_segscan_highly_segmented"),
    "test_segscan_highly_segmented2": (
        "test_segscan",
        "test_segscan_highly_segmented2",
    ),
    "highly_segmented3": ("test_segscan", "highly_segmented3"),
    "test_segscan_highly_segmented3": (
        "test_segscan",
        "test_segscan_highly_segmented3",
    ),
    "test_segscan_excl_cuda_bench": ("test_segscan", "test_segscan_excl_cuda_bench"),
    "test_segscan_incl_cuda_bench": ("test_segscan", "test_segscan_incl_cuda_bench"),
    "test_segscan_cpu_bench": ("test_segscan", "test_segscan_cpu_bench"),
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
