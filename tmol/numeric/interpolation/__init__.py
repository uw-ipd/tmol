_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "interpolate_t": ("cubic_hermite_polynomial", "interpolate_t"),
    "interpolate": ("cubic_hermite_polynomial", "interpolate"),
    "interpolate_dt": ("cubic_hermite_polynomial", "interpolate_dt"),
    "interpolate_dx": ("cubic_hermite_polynomial", "interpolate_dx"),
    "interpolate_to_zero_t": ("cubic_hermite_polynomial", "interpolate_to_zero_t"),
    "interpolate_to_zero": ("cubic_hermite_polynomial", "interpolate_to_zero"),
    "interpolate_to_zero_dt": ("cubic_hermite_polynomial", "interpolate_to_zero_dt"),
    "interpolate_to_zero_dx": ("cubic_hermite_polynomial", "interpolate_to_zero_dx"),
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
