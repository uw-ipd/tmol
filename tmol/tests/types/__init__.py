_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "validation_examples": ("test_array", "validation_examples"),
    "test_array_validation": ("test_array", "test_array_validation"),
    "packed_dtype": ("test_array", "packed_dtype"),
    "incompatible_dtype": ("test_array", "incompatible_dtype"),
    "converstion_examples": ("test_array", "converstion_examples"),
    "test_array_conversion": ("test_array", "test_array_conversion"),
    "ValidateObj": ("test_attrs", "ValidateObj"),
    "test_validate_attrs": ("test_attrs", "test_validate_attrs"),
    "test_set_post_init": ("test_attrs", "test_set_post_init"),
    "ConvertObj": ("test_attrs", "ConvertObj"),
    "test_convert_attrs": ("test_attrs", "test_convert_attrs"),
    "BufferModule": ("test_attrs_nn_module", "BufferModule"),
    "test_buffers": ("test_attrs_nn_module", "test_buffers"),
    "f": ("test_functional", "f"),
    "int_func": ("test_functional", "int_func"),
    "union_func": ("test_functional", "union_func"),
    "anytuple_func": ("test_functional", "anytuple_func"),
    "nest_tuple_func": ("test_functional", "nest_tuple_func"),
    "tuple_func": ("test_functional", "tuple_func"),
    "ellipsis_tuple_func": ("test_functional", "ellipsis_tuple_func"),
    "str_func": ("test_functional", "str_func"),
    "array_func": ("test_functional", "array_func"),
    "union_array_func": ("test_functional", "union_array_func"),
    "tuple_array_func": ("test_functional", "tuple_array_func"),
    "list_int_func": ("test_functional", "list_int_func"),
    "list_func": ("test_functional", "list_func"),
    "list_union_func": ("test_functional", "list_union_func"),
    "validate_examples": ("test_functional", "validate_examples"),
    "test_func_validation": ("test_functional", "test_func_validation"),
    "int_cfunc": ("test_functional", "int_cfunc"),
    "str_cfunc": ("test_functional", "str_cfunc"),
    "array_cfunc": ("test_functional", "array_cfunc"),
    "union_cfunc": ("test_functional", "union_cfunc"),
    "tuple_cfunc": ("test_functional", "tuple_cfunc"),
    "convert_examples": ("test_functional", "convert_examples"),
    "test_func_conversion": ("test_functional", "test_func_conversion"),
    "test_return_annotation": ("test_functional", "test_return_annotation"),
    "testShape": ("test_shape", "testShape"),
    "test_attr_checking": ("test_tensor", "test_attr_checking"),
    "SubGroup": ("test_tensor", "SubGroup"),
    "MultiGroup": ("test_tensor", "MultiGroup"),
    "test_nested_group": ("test_tensor", "test_nested_group"),
    "test_tensor_group_reshape": ("test_tensor", "test_tensor_group_reshape"),
    "test_tensor_group_invalid_reshape": (
        "test_tensor",
        "test_tensor_group_invalid_reshape",
    ),
    "test_tensorgroup_smoke": ("test_tensor", "test_tensorgroup_smoke"),
    "test_tensorgroup_cat": ("test_tensor", "test_tensorgroup_cat"),
    "test_tensorgroup_to_dtypes": ("test_tensor", "test_tensorgroup_to_dtypes"),
    "test_tensorgroup_to_device": ("test_tensor", "test_tensorgroup_to_device"),
    "invalid_dtypes": ("test_torch", "invalid_dtypes"),
    "test_invalid_dtype": ("test_torch", "test_invalid_dtype"),
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
