import inspect
from decorator import decorate
import typing
from tmol.extern.typeguard import typechecked

from .converters import get_converter

validate_args = typechecked


def convert_args(f):
    f._signature = inspect.signature(f)
    f._converters = {
        n: get_converter(v)
        for n, v in typing.get_type_hints(f).items()
    }

    def convert_f(f, *args, **kwargs):
        bound = f._signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for n, val in bound.arguments.items():
            converter = f._converters.get(n, None)
            if converter:
                try:
                    bound.arguments[n] = converter(val)
                except Exception as vexec:
                    raise TypeError(
                        f"Invalid argument: {n} {vexec!s}"
                    ) from vexec

        retval = f(*bound.args, **bound.kwargs)

        converter = f._converters.get("return", None)
        if converter:
            try:
                retval = converter(retval)
            except Exception as vexec:
                raise TypeError(f"Invalid return value: {vexec!s}") from vexec
        return retval

    return decorate(f, convert_f)
