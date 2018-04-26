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


get_converter.register(typing._Union)(validate_convert)
get_converter.register(typing.TupleMeta)(validate_convert)
