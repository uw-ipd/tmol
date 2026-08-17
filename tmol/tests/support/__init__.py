_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "pyrosetta": ("rosetta", "pyrosetta"),
    "rosetta_database": ("rosetta", "rosetta_database"),
    "pyrosetta_available": ("rosetta", "pyrosetta_available"),
    "requires_pyrosetta": ("rosetta", "requires_pyrosetta"),
    "rosetta_database_available": ("rosetta", "rosetta_database_available"),
    "requires_rosetta_database": ("rosetta", "requires_rosetta_database"),
    "test_hbond_param_import": ("test_database_converters", "test_hbond_param_import"),
    "test_rama_table_read": ("test_database_converters", "test_rama_table_read"),
    "test_bbdep_omega_table_read": (
        "test_database_converters",
        "test_bbdep_omega_table_read",
    ),
    "test_dunbrack_table_read": (
        "test_database_converters",
        "test_dunbrack_table_read",
    ),
    "test_fixture": ("test_pyrosetta_import", "test_fixture"),
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
