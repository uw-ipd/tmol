"""Generic type validator functions."""

from functools import singledispatch

from typing_inspect import is_tuple_type, is_union_type
import toolz

_validators = []


@singledispatch
def get_validator(type_annotation):
    for pred, val in _validators:
        if pred(type_annotation):
            return val(type_annotation)
    else:
        return validate_isinstance(type_annotation)


@toolz.curry
def validate_tuple(tup, value):
    if not isinstance(value, tuple):
        raise TypeError(f"expected {tup}, received {type(value)!r}")

    if tup.__args__ and tup.__args__[-1] == Ellipsis:
        vval = get_validator(tup.__args__[0])
        for v in value:
            vval(v)
    elif tup.__args__:
        if len(tup.__args__) != len(value):
            raise ValueError(f"expected {tup}, received invalid length: {len(value)}")

        for tt, v in zip(tup.__args__, value):
            get_validator(tt)(v)


@toolz.curry
def validate_union(union, value):
    assert union.__args__

    last_ex = None

    for ut in union.__args__:
        validator = get_validator(ut)

        try:
            validator(value)
            return
        except (ValueError, TypeError) as ex:
            last_ex = ex

    raise TypeError(f"expected {union}, received {type(value)!r}") from last_ex


@toolz.curry
def validate_isinstance(type_annotation, value):
    if not isinstance(value, type_annotation):
        raise TypeError(f"expected {type_annotation}, received {type(value)!r}")


def register_validator(type_predicate, validator):
    _validators.append((type_predicate, validator))


register_validator(is_tuple_type, validate_tuple)
register_validator(is_union_type, validate_union)
