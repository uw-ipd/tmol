import inspect
from decorator import decorate
from functools import singledispatch

import typing
import toolz


@toolz.curry
def validate_isinstance(type_annotation, value):
    if not isinstance(value, type_annotation):
        raise TypeError(
            f"expected {type_annotation}, received {type(value)!r}"
        )


@singledispatch
def get_validator(type_annotation):
    return validate_isinstance(type_annotation)


def validate_args(f):
    f._signature = inspect.signature(f)
    f._validators = {
        n: get_validator(v)
        for n, v in typing.get_type_hints(f).items()
    }

    def validate_f(f, *args, **kwargs):
        bound = f._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for n, v in f._validators.items():
            try:
                v(bound.arguments[n])
            except Exception as vexec:
                raise TypeError(f"Invalid argument: {n}") from vexec

        return f(*args, **kwargs)

    return decorate(f, validate_f)


@toolz.curry
def constructor_convert(type_annotation, value):
    if isinstance(value, type_annotation):
        return value
    else:
        return type_annotation(value)


@singledispatch
def get_converter(type_annotation):
    return constructor_convert(type_annotation)


def convert_args(f):
    f._signature = inspect.signature(f)
    f._converters = {
        n: get_converter(v)
        for n, v in typing.get_type_hints(f).items()
    }

    def convert_f(f, *args, **kwargs):
        bound = f._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for n, v in f._converters.items():
            try:
                bound.arguments[n] = v(bound.arguments[n])
            except Exception as vexec:
                raise TypeError(f"Invalid argument: {n}") from vexec

        return f(*bound.args, **bound.kwargs)

    return decorate(f, convert_f)
