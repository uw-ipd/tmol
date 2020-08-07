"""Runtime type validation and conversion."""

import inspect
from decorator import decorate
import typing

from .validators import get_validator
from .converters import get_converter


def validate_args(f):
    f._signature = inspect.signature(f)
    f._validators = {n: get_validator(v) for n, v in typing.get_type_hints(f).items()}

    def validate_f(f, *args, **kwargs):
        bound = f._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for n, val in bound.arguments.items():
            validator = f._validators.get(n, None)
            if validator:
                try:
                    validator(val)
                except Exception as vexec:
                    raise TypeError(f"Invalid argument: {n}") from vexec

        retval = f(*args, **kwargs)

        validator = f._validators.get("return", None)
        if validator:
            try:
                validator(retval)
            except Exception as vexec:
                raise TypeError("Invalid return value") from vexec

        return retval

    return decorate(f, validate_f)


def convert_args(f):
    f._signature = inspect.signature(f)
    f._converters = {n: get_converter(v) for n, v in typing.get_type_hints(f).items()}

    def convert_f(f, *args, **kwargs):
        bound = f._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for n, val in bound.arguments.items():
            converter = f._converters.get(n, None)
            if converter:
                try:
                    bound.arguments[n] = converter(val)
                except Exception as vexec:
                    raise TypeError(f"Invalid argument: {n}") from vexec

        retval = f(*bound.args, **bound.kwargs)

        converter = f._converters.get("return", None)
        if converter:
            try:
                retval = converter(retval)
            except Exception as vexec:
                raise TypeError("Invalid return value") from vexec
        return retval

    return decorate(f, convert_f)
