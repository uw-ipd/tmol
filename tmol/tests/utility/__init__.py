_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_attr_mapping": ("test_attr", "test_attr_mapping"),
    "test_categorical_conversion": ("test_categorical", "test_categorical_conversion"),
    "test_flag_enum": ("test_categorical", "test_flag_enum"),
    "test_flatten": ("test_dicttoolz", "test_flatten"),
    "Setup": ("test_mixins", "Setup"),
    "test_cooperative_factory_function_update": (
        "test_mixins",
        "test_cooperative_factory_function_update",
    ),
    "test_cooperative_factory_kwargs": (
        "test_mixins",
        "test_cooperative_factory_kwargs",
    ),
    "test_cooperative_factory_dispatch": (
        "test_mixins",
        "test_cooperative_factory_dispatch",
    ),
    "test_angle_parsing": ("test_units", "test_angle_parsing"),
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
