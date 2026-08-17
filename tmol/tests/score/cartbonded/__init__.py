_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_smoke": ("test_cartbonded_energy_term", "test_smoke"),
    "test_annotate_twice": ("test_cartbonded_energy_term", "test_annotate_twice"),
    "test_annotate_restypes": ("test_cartbonded_energy_term", "test_annotate_restypes"),
    "test_hack_cartbonded_params": (
        "test_cartbonded_energy_term",
        "test_hack_cartbonded_params",
    ),
    "TestCartBondedEnergyTerm": (
        "test_cartbonded_energy_term",
        "TestCartBondedEnergyTerm",
    ),
    "GROUPS": ("test_cartbonded_param_reachability", "GROUPS"),
    "INERT_ATOMS": ("test_cartbonded_param_reachability", "INERT_ATOMS"),
    "test_cross_marked_atoms_form_a_trailing_run": (
        "test_cartbonded_param_reachability",
        "test_cross_marked_atoms_form_a_trailing_run",
    ),
    "test_no_cross_marked_atoms_in_improper_params": (
        "test_cartbonded_param_reachability",
        "test_no_cross_marked_atoms_in_improper_params",
    ),
    "test_wildcard_intra_rows_are_realizable": (
        "test_cartbonded_param_reachability",
        "test_wildcard_intra_rows_are_realizable",
    ),
    "test_cross_rows_are_not_realizable_intra": (
        "test_cartbonded_param_reachability",
        "test_cross_rows_are_not_realizable_intra",
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
