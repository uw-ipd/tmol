_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "lbfgs_two_loop": ("lbfgs_armijo", "lbfgs_two_loop"),
    "armijo_linesearch_segmented": ("lbfgs_armijo", "armijo_linesearch_segmented"),
    "LBFGS_Armijo": ("lbfgs_armijo", "LBFGS_Armijo"),
    "build_kinforest_network": ("minimizers", "build_kinforest_network"),
    "run_min": ("minimizers", "run_min"),
    "run_kin_min": ("minimizers", "run_kin_min"),
    "run_cart_min": ("minimizers", "run_cart_min"),
    "CartesianSfxnNetwork": ("sfxn_modules", "CartesianSfxnNetwork"),
    "KinForestSfxnNetwork": ("sfxn_modules", "KinForestSfxnNetwork"),
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
