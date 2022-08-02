"""Generic type validator functions."""

from functools import singledispatch

from typing import List, TypeVar
from typing_inspect import is_tuple_type, is_union_type
import toolz

_validators = []


def is_list_type(tp):
    """Test if the type is a generic list type, including subclasses excluding
    non-generic classes.
    Examples::
        is_list_type(int) == False
        is_list_type(list) == False
        is_list_type(List) == True
        is_list_type(List[int]) == True
        is_list_type(List[str, int]) == True
        class MyClass(List[str, int]):
            ...
        is_tuple_type(MyClass) == True
    For more general tests use issubclass(..., list), for more precise test
    (excluding subclasses) use::
        get_origin(tp) is list  # Tuple prior to Python 3.7
    """

    from typing_inspect import NEW_TYPING

    if NEW_TYPING:
        import sys
        from typing import Generic, _GenericAlias

        if sys.version_info[:3] >= (3, 9, 0):
            from typing import _SpecialGenericAlias
            from types import GenericAlias

            typingGenericAlias = (_GenericAlias, _SpecialGenericAlias, GenericAlias)
        else:
            typingGenericAlias = (_GenericAlias,)

        return (
            tp is List
            or isinstance(tp, typingGenericAlias)
            and tp.__origin__ is list
            or isinstance(tp, type)
            and issubclass(tp, Generic)
            and issubclass(tp, list)
        )

    # only attempt to import if we have an old version of python?
    # is this needed? We are targetting tmol for python3.7+
    from typing import ListMeta

    return type(tp) is ListMeta


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
def validate_list(lst, value):
    """Test if a given value matches the List type in the type hints:
    A list may either be of a uniform type, e.g. "List[int]", or may have
    no specified type, and thus be of any time, e.g. "List". In the first
    case, the single type may be a Union, e.g. "List[Union[int, str]]".

    validate_list(List[int], [5]) == True
    validate_list(List[int], [5, 4, 3]) == True
    validate_list(List[int], [5, "thumb"]) == False
    validate_list(List, 5) == False
    validate_list(List, []) == True
    validate_list(List, [5, "thumb"]) == True
    """
    if not isinstance(value, list):
        raise TypeError(f"expected {lst}, received {type(value)!r}")

    if type(lst.__args__[0]) != TypeVar:
        # accept List[X] as a list with as many elements of type X as you want
        for i, v in enumerate(value):
            try:
                get_validator(lst.__args__[0])(v)
            except TypeError as err:
                raise TypeError(
                    f"Failed to validate {lst}: {i}th argument error: {err}"
                ) from err


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
register_validator(is_list_type, validate_list)
register_validator(is_union_type, validate_union)
