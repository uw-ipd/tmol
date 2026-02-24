"""Generic type conversion functions."""

from functools import singledispatch

from typing_inspect import is_tuple_type, is_union_type
import toolz

from . import validators

_converters = []


@singledispatch
def get_converter(type_annotation):
    for pred, conv in _converters:
        if pred(type_annotation):
            return conv(type_annotation)

    return constructor_convert(type_annotation)


@toolz.curry
def constructor_convert(type_annotation, value):
    if isinstance(value, type_annotation):
        return value
    else:
        return type_annotation(value)


@toolz.curry
def validate_convert(type_annotation, value):
    validators.get_validator(type_annotation)(value)

    return value


@toolz.curry
def union_convert(union_annotation, value):
    for subtype in union_annotation.__args__:
        try:
            validators.get_validator(subtype)(value)
            return value
        except (TypeError, ValueError):
            pass

    errors = []
    for subtype in union_annotation.__args__:
        try:
            result = get_converter(subtype)(value)
            return result
        except (TypeError, ValueError) as ex:
            errors.append(ex)

    raise TypeError(
        "Unable to convert to any union subtype: {union_annotaion} value: {value}"
    )


def register_converter(type_predicate, converter):
    _converters.append((type_predicate, converter))


register_converter(is_union_type, union_convert)
register_converter(is_tuple_type, validate_convert)
