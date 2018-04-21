from functools import singledispatch

import typing
import toolz


@singledispatch
def get_validator(type_annotation):
    return validate_isinstance(type_annotation)


@get_validator.register(typing._Union)
@toolz.curry
def validate_union(union, value):
    assert union.__args__

    if not isinstance(value, union.__args__):
        raise TypeError(f"expected {union}, received {type(value)!r}")


@toolz.curry
def validate_isinstance(type_annotation, value):
    if not isinstance(value, type_annotation):
        raise TypeError(
            f"expected {type_annotation}, received {type(value)!r}"
        )
