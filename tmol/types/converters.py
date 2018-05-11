from functools import singledispatch

import typing
import toolz

from . import validators


@singledispatch
def get_converter(type_annotation):
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


get_converter.register(typing._Union)(union_convert)
get_converter.register(typing.TupleMeta)(validate_convert)
