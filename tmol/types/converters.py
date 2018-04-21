from functools import singledispatch

import typing
import toolz


@singledispatch
def get_converter(type_annotation):
    return constructor_convert(type_annotation)


@toolz.curry
def constructor_convert(type_annotation, value):
    if isinstance(value, type_annotation):
        return value
    else:
        return type_annotation(value)


@get_converter.register(typing._Union)
@toolz.curry
def validate_union(union, value):
    assert union.__args__

    if not isinstance(value, union.__args__):
        raise TypeError(
            f"unable to convert to union type. expected {union}, received {type(value)!r}"
        )
