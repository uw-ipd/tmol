_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "bonds_and_bond_ranges": ("all_bonds", "bonds_and_bond_ranges"),
    "MAX_SIG_BOND_SEPARATION": ("constants", "MAX_SIG_BOND_SEPARATION"),
    "MAX_PATHS_FROM_CONNECTION": ("constants", "MAX_PATHS_FROM_CONNECTION"),
    "eye4": ("ideal_coords", "eye4"),
    "normalize": ("ideal_coords", "normalize"),
    "frame_from_coords": ("ideal_coords", "frame_from_coords"),
    "rot_x": ("ideal_coords", "rot_x"),
    "rot_z": ("ideal_coords", "rot_z"),
    "trans_z": ("ideal_coords", "trans_z"),
    "build_coords_from_icoors": ("ideal_coords", "build_coords_from_icoors"),
    "build_ideal_coords": ("ideal_coords", "build_ideal_coords"),
    "AtomIndex": ("restypes", "AtomIndex"),
    "ConnectionIndex": ("restypes", "ConnectionIndex"),
    "BondCount": ("restypes", "BondCount"),
    "BondType": ("restypes", "BondType"),
    "BOND_TYPE_FROM_STR": ("restypes", "BOND_TYPE_FROM_STR"),
    "UnresolvedAtomID": ("restypes", "UnresolvedAtomID"),
    "uaid_t": ("restypes", "uaid_t"),
    "ResName3": ("restypes", "ResName3"),
    "IcoorIndex": ("restypes", "IcoorIndex"),
    "three2one": ("restypes", "three2one"),
    "one2three": ("restypes", "one2three"),
    "RefinedResidueType": ("restypes", "RefinedResidueType"),
    "ResidueTypeSet": ("restypes", "ResidueTypeSet"),
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
