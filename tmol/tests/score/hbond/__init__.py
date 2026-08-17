_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_hbond_dep_term_annotate_block_types_smoke": (
        "test_hbond_dependent_term",
        "test_hbond_dep_term_annotate_block_types_smoke",
    ),
    "test_hbond_dep_term_annotate_packed_block_types_smoke": (
        "test_hbond_dependent_term",
        "test_hbond_dep_term_annotate_packed_block_types_smoke",
    ),
    "test_hbond_dep_term_setup_packed_block_types": (
        "test_hbond_dependent_term",
        "test_hbond_dep_term_setup_packed_block_types",
    ),
    "test_hbond_dep_term_setup_ser_block_type": (
        "test_hbond_dependent_term",
        "test_hbond_dep_term_setup_ser_block_type",
    ),
    "test_smoke": ("test_hbond_energy_term", "test_smoke"),
    "test_hbond_in_sfxn": ("test_hbond_energy_term", "test_hbond_in_sfxn"),
    "test_annotate_restypes": ("test_hbond_energy_term", "test_annotate_restypes"),
    "test_whole_pose_scoring_module_smoke": (
        "test_hbond_energy_term",
        "test_whole_pose_scoring_module_smoke",
    ),
    "TestHBondEnergyTerm": ("test_hbond_energy_term", "TestHBondEnergyTerm"),
    "test_every_donor_acceptor_pair_is_parameterized": (
        "test_hbond_pair_coverage",
        "test_every_donor_acceptor_pair_is_parameterized",
    ),
    "test_resolved_pair_params_are_finite": (
        "test_hbond_pair_coverage",
        "test_resolved_pair_params_are_finite",
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
