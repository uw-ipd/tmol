_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "lj_score_V": ("compiled", "lj_score_V"),
    "lj_score_V_dV": ("compiled", "lj_score_V_dV"),
    "lk_isotropic_score_V": ("compiled", "lk_isotropic_score_V"),
    "lk_isotropic_score_V_dV": ("compiled", "lk_isotropic_score_V_dV"),
    "lj_sigma": ("compiled", "lj_sigma"),
    "vdw_V": ("compiled", "vdw_V"),
    "vdw_V_dV": ("compiled", "vdw_V_dV"),
    "f_desolv_V": ("compiled", "f_desolv_V"),
    "f_desolv_V_dV": ("compiled", "f_desolv_V_dV"),
    "LJScore": ("compiled", "LJScore"),
    "LKScore": ("compiled", "LKScore"),
    "params": ("test_compiled_lj_potential", "params"),
    "parametrize_atom_pairs": ("test_compiled_lj_potential", "parametrize_atom_pairs"),
    "test_lj_gradcheck": ("test_compiled_lj_potential", "test_lj_gradcheck"),
    "test_lj_spotcheck": ("test_compiled_lj_potential", "test_lj_spotcheck"),
    "CARBON_LK_TYPES": ("test_compiled_lk_isotropic_potential", "CARBON_LK_TYPES"),
    "test_lk_isotropic_gradcheck": (
        "test_compiled_lk_isotropic_potential",
        "test_lk_isotropic_gradcheck",
    ),
    "test_lk_spotcheck": ("test_compiled_lk_isotropic_potential", "test_lk_spotcheck"),
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
