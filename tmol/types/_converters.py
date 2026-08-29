"""Generic type conversion functions."""

from functools import singledispatch

from typing_inspect import is_tuple_type, is_union_type
import toolz

from . import _validators as validators

_converters = []


@singledispatch
def get_converter(type_annotation):
    """Return the registered value converter for a type annotation."""
    for pred, conv in _converters:
        if pred(type_annotation):
            return conv(type_annotation)

    return constructor_convert(type_annotation)


@toolz.curry
def constructor_convert(type_annotation, value):
    """Convert a value by calling the annotated type when necessary."""
    if isinstance(value, type_annotation):
        return value
    return type_annotation(value)


@toolz.curry
def validate_convert(type_annotation, value):
    """Validate a value against an annotation and return it unchanged."""
    validators.get_validator(type_annotation)(value)

    return value


@toolz.curry
def union_convert(union_annotation, value):
    """Convert a value using the first compatible member of a union."""
    for subtype in union_annotation.__args__:
        try:
            validators.get_validator(subtype)(value)
            return value
        except (TypeError, ValueError):
            pass

    errors = []
    for subtype in union_annotation.__args__:
        try:
            return get_converter(subtype)(value)
        except (TypeError, ValueError) as ex:
            errors.append(ex)

    raise TypeError(
        f"Unable to convert to any union subtype: {union_annotation} value: {value!r}"
    )


def register_converter(type_predicate, converter):
    """Register a converter factory for annotations matching a predicate."""
    _converters.append((type_predicate, converter))


register_converter(is_union_type, union_convert)
register_converter(is_tuple_type, validate_convert)
