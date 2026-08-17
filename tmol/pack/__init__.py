_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_missing_sidechains": (
        "build_missing_sidechains",
        "build_missing_sidechains",
    ),
    "PackerEnergyTables": ("datatypes", "PackerEnergyTables"),
    "impose_top_rotamer_assignments": (
        "impose_rotamers",
        "impose_top_rotamer_assignments",
    ),
    "pack_rotamers": ("pack_rotamers", "pack_rotamers"),
    "set_compare": ("packer_task", "set_compare"),
    "PackerPalleteAnnotation": ("packer_task", "PackerPalleteAnnotation"),
    "PackerPalette": ("packer_task", "PackerPalette"),
    "PackerTask": ("packer_task", "PackerTask"),
    "SetPackerTask": ("packer_task", "SetPackerTask"),
    "run_simulated_annealing": ("simulated_annealing", "run_simulated_annealing"),
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
