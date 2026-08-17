_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_na_torsion_benchmark": (
        "test_na_torsion_benchmark",
        "test_na_torsion_benchmark",
    ),
    "test_smoke": ("test_na_torsion_energy_term", "test_smoke"),
    "test_all_na_block_types_are_scoreable": (
        "test_na_torsion_energy_term",
        "test_all_na_block_types_are_scoreable",
    ),
    "test_scores_na_and_ignores_protein": (
        "test_na_torsion_energy_term",
        "test_scores_na_and_ignores_protein",
    ),
    "test_stacked_poses_scale_linearly": (
        "test_na_torsion_energy_term",
        "test_stacked_poses_scale_linearly",
    ),
    "test_subterms_sum_to_the_totals": (
        "test_na_torsion_energy_term",
        "test_subterms_sum_to_the_totals",
    ),
    "test_rotamer_energies_match_the_pose_energies": (
        "test_na_torsion_energy_term",
        "test_rotamer_energies_match_the_pose_energies",
    ),
    "test_gradcheck": ("test_na_torsion_energy_term", "test_gradcheck"),
    "test_wrap_degrees_range": (
        "test_na_torsion_energy_term",
        "test_wrap_degrees_range",
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
