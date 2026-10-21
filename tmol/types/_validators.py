"""Generic type validator functions."""

from functools import singledispatch

from typing import List, get_args, get_origin
from typing_inspect import is_tuple_type, is_union_type
import toolz

_validators = []


def _is_list_type(tp):
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

    return tp is List or get_origin(tp) is list


@singledispatch
def _get_validator(type_annotation):
    """Return the registered runtime validator for a type annotation."""
    for pred, val in _validators:
        if pred(type_annotation):
            return val(type_annotation)
    return _validate_isinstance(type_annotation)


@toolz.curry
def _validate_tuple(tup, value):
    if not isinstance(value, tuple):
        raise TypeError(f"expected {tup}, received {type(value)!r}")

    args = get_args(tup)

    if args and args[-1] == Ellipsis:
        vval = _get_validator(args[0])
        for v in value:
            vval(v)
    elif args:
        if len(tup.__args__) != len(value):
            raise ValueError(f"expected {tup}, received invalid length: {len(value)}")

        for tt, v in zip(tup.__args__, value):
            _get_validator(tt)(v)


@toolz.curry
def _validate_list(lst, value):
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

    args = get_args(lst)

    if args:
        # accept List[X] as a list with as many elements of type X as you want
        for i, v in enumerate(value):
            try:
                _get_validator(args[0])(v)
            except TypeError as err:
                raise TypeError(
                    f"Failed to validate {lst}: {i}th argument error: {err}"
                ) from err


@toolz.curry
def _validate_union(union, value):
    assert union.__args__

    for ut in union.__args__:
        try:
            _get_validator(ut)(value)
            return
        except (ValueError, TypeError):
            pass

    raise TypeError(f"expected {union}, received {type(value)!r}")


@toolz.curry
def _validate_isinstance(type_annotation, value):
    """Require a value to be an instance of the annotated type."""
    if not isinstance(value, type_annotation):
        raise TypeError(f"expected {type_annotation}, received {type(value)!r}")


def _register_validator(type_predicate, validator):
    """Register a validator factory for annotations matching a predicate."""
    _validators.append((type_predicate, validator))


_register_validator(is_tuple_type, _validate_tuple)
_register_validator(_is_list_type, _validate_list)
_register_validator(is_union_type, _validate_union)
