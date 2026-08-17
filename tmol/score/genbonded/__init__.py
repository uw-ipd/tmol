_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "MAX_HIER_DEPTH": ("genbonded_energy_term", "MAX_HIER_DEPTH"),
    "BOND_CHAR_TO_INT": ("genbonded_energy_term", "BOND_CHAR_TO_INT"),
    "BOND_TYPE_TO_CHAR": ("genbonded_energy_term", "BOND_TYPE_TO_CHAR"),
    "GB_WILDCARD_BOND_INT": ("genbonded_energy_term", "GB_WILDCARD_BOND_INT"),
    "GenBondedEnergyTerm": ("genbonded_energy_term", "GenBondedEnergyTerm"),
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
