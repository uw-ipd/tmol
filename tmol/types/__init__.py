"""Support for runtime type validation and conversion."""

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Casting": ("array", "Casting"),
    "NDArray": ("array", "NDArray"),
    "ConvertAttrs": ("attrs", "ConvertAttrs"),
    "ValidateAttrs": ("attrs", "ValidateAttrs"),
    "get_converter": ("converters", "get_converter"),
    "constructor_convert": ("converters", "constructor_convert"),
    "validate_convert": ("converters", "validate_convert"),
    "union_convert": ("converters", "union_convert"),
    "register_converter": ("converters", "register_converter"),
    "validate_args": ("functional", "validate_args"),
    "convert_args": ("functional", "convert_args"),
    "Dim": ("shape", "Dim"),
    "Shape": ("shape", "Shape"),
    "SubscriptableType": ("subscriptable", "SubscriptableType"),
    "TensorGroup": ("tensor", "TensorGroup"),
    "cat": ("tensor", "cat"),
    "torch_dtype": ("torch", "torch_dtype"),
    "like_kwargs": ("torch", "like_kwargs"),
    "Tensor": ("torch", "Tensor"),
    "is_list_type": ("validators", "is_list_type"),
    "get_validator": ("validators", "get_validator"),
    "validate_tuple": ("validators", "validate_tuple"),
    "validate_list": ("validators", "validate_list"),
    "validate_union": ("validators", "validate_union"),
    "validate_isinstance": ("validators", "validate_isinstance"),
    "register_validator": ("validators", "register_validator"),
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
