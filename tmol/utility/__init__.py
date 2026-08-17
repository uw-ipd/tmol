from toolz import first

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "bind_to_args": ("args", "bind_to_args"),
    "ignore_unused_kwargs": ("args", "ignore_unused_kwargs"),
    "AttrMapping": ("attr", "AttrMapping"),
    "AttrMutableMapping": ("attr", "AttrMutableMapping"),
    "AutoNumber": ("auto_number", "AutoNumber"),
    "get_all_segment_positions": ("biotite_util", "get_all_segment_positions"),
    "get_all_residue_positions": ("biotite_util", "get_all_residue_positions"),
    "enum_val_catdtype": ("categorical", "enum_val_catdtype"),
    "enum_name_catdtype": ("categorical", "enum_name_catdtype"),
    "vals_to_val_cat": ("categorical", "vals_to_val_cat"),
    "vals_to_name_cat": ("categorical", "vals_to_name_cat"),
    "names_to_name_cat": ("categorical", "names_to_name_cat"),
    "names_to_val_cat": ("categorical", "names_to_val_cat"),
    "get_torch_version": ("cpp_extension", "get_torch_version"),
    "cuda_if_available": ("cpp_extension", "cuda_if_available"),
    "load": ("cpp_extension", "load"),
    "load_inline": ("cpp_extension", "load_inline"),
    "relpaths": ("cpp_extension", "relpaths"),
    "modulename": ("cpp_extension", "modulename"),
    "exclusive_cumsum": ("cumsum", "exclusive_cumsum"),
    "exclusive_cumsum1d": ("cumsum", "exclusive_cumsum1d"),
    "exclusive_cumsum2d": ("cumsum", "exclusive_cumsum2d"),
    "exclusive_cumsum2d_w_totals": ("cumsum", "exclusive_cumsum2d_w_totals"),
    "resolve_device": ("device", "resolve_device"),
    "items": ("dicttoolz", "items"),
    "keys": ("dicttoolz", "keys"),
    "vals": ("dicttoolz", "vals"),
    "flat_items": ("dicttoolz", "flat_items"),
    "unflatten": ("dicttoolz", "unflatten"),
    "update_inplace": ("dicttoolz", "update_inplace"),
    "classlogger_for": ("log", "classlogger_for"),
    "logger_for_class": ("log", "logger_for_class"),
    "ClassLogger": ("log", "ClassLogger"),
    "LoggerMixin": ("log", "LoggerMixin"),
    "QualifiedName": ("mixins", "QualifiedName"),
    "qualified_name": ("mixins", "qualified_name"),
    "gather_superclass_properies": ("mixins", "gather_superclass_properies"),
    "cooperative_superclass_factory": ("mixins", "cooperative_superclass_factory"),
    "torch_cuda_array_interface": ("numba", "torch_cuda_array_interface"),
    "nvtx_range": ("nvtx", "nvtx_range"),
    "ureg": ("units", "ureg"),
    "u": ("units", "u"),
    "parse_angle": ("units", "parse_angle"),
    "parse_bond_angle": ("units", "parse_bond_angle"),
    "parse_dihedral_angle": ("units", "parse_dihedral_angle"),
    "Angle": ("units", "Angle"),
    "BondAngle": ("units", "BondAngle"),
    "DihedralAngle": ("units", "DihedralAngle"),
    "_signature": ("args", "_signature"),
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


def unique_val(vals):
    """Extract a single, unique value from a collection of values."""
    return just_one(set(vals))


def just_one(vals):
    """Extract a single value from a length one collection of values."""
    assert len(vals) == 1
    return first(vals)
