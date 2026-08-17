_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "hbond_score_V_dV": ("compiled", "hbond_score_V_dV"),
    "AH_dist_V_dV": ("compiled", "AH_dist_V_dV"),
    "AHD_angle_V_dV": ("compiled", "AHD_angle_V_dV"),
    "BAH_angle_V_dV": ("compiled", "BAH_angle_V_dV"),
    "sp2chi_energy_V_dV": ("compiled", "sp2chi_energy_V_dV"),
    "poly_from_lists": ("test_potentials", "poly_from_lists"),
    "merge_polys": ("test_potentials", "merge_polys"),
    "compiled": ("test_potentials", "compiled"),
    "sp2_params": ("test_potentials", "sp2_params"),
    "sp3_params": ("test_potentials", "sp3_params"),
    "ring_params": ("test_potentials", "ring_params"),
    "hbsc_subset": ("test_potentials", "hbsc_subset"),
    "test_hbond_point_scores": ("test_potentials", "test_hbond_point_scores"),
    "test_hbond_point_scores_gradcheck": (
        "test_potentials",
        "test_hbond_point_scores_gradcheck",
    ),
    "test_AH_dist_gradcheck": ("test_potentials", "test_AH_dist_gradcheck"),
    "test_AHD_angle_gradcheck": ("test_potentials", "test_AHD_angle_gradcheck"),
    "test_BAH_angle_gradcheck": ("test_potentials", "test_BAH_angle_gradcheck"),
    "test_sp2_chi_energy_gradcheck": (
        "test_potentials",
        "test_sp2_chi_energy_gradcheck",
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
