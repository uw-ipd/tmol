"""Generic type conversion functions."""

from functools import singledispatch

from typing_inspect import is_tuple_type, is_union_type
import toolz

from . import _validators as validators

_converters = []


@singledispatch
def _get_converter(type_annotation):
    """Return the registered value converter for a type annotation."""
    for pred, conv in _converters:
        if pred(type_annotation):
            return conv(type_annotation)

    return _constructor_convert(type_annotation)


@toolz.curry
def _constructor_convert(type_annotation, value):
    """Convert a value by calling the annotated type when necessary."""
    if isinstance(value, type_annotation):
        return value
    return type_annotation(value)


@toolz.curry
def _validate_convert(type_annotation, value):
    """Validate a value against an annotation and return it unchanged."""
    validators._get_validator(type_annotation)(value)

    return value


@toolz.curry
def _union_convert(union_annotation, value):
    """Convert a value using the first compatible member of a union."""
    for subtype in union_annotation.__args__:
        try:
            validators._get_validator(subtype)(value)
            return value
        except (TypeError, ValueError):
            pass

    for subtype in union_annotation.__args__:
        try:
            return _get_converter(subtype)(value)
        except (TypeError, ValueError):
            pass

    raise TypeError(
        f"Unable to convert to any union subtype: {union_annotation} value: {value!r}"
    )


def _register_converter(type_predicate, converter):
    """Register a converter factory for annotations matching a predicate."""
    _converters.append((type_predicate, converter))


_register_converter(is_union_type, _union_convert)
_register_converter(is_tuple_type, _validate_convert)
