_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_2d_bspline": ("test_bspline", "test_2d_bspline"),
    "test_2d_bspline_off_grid": ("test_bspline", "test_2d_bspline_off_grid"),
    "test_2d_bspline_off_grid_at_edges": (
        "test_bspline",
        "test_2d_bspline_off_grid_at_edges",
    ),
    "test_2d_bspline_not_square": ("test_bspline", "test_2d_bspline_not_square"),
    "test_3d_bspline": ("test_bspline", "test_3d_bspline"),
    "test_3d_bspline_not_square": ("test_bspline", "test_3d_bspline_not_square"),
    "test_4d_bspline": ("test_bspline", "test_4d_bspline"),
    "test_coord_dihedrals": ("test_dihedrals", "test_coord_dihedrals"),
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
