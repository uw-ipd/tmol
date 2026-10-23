import gc
import weakref
from contextlib import contextmanager

import numpy
import pytest

import typing
from typing import Union, Tuple, List

from tmol.types._functional import _validate_args, _convert_args
from tmol.types._array import NDArray
from tmol.types._validators import _get_validator
from tmol.types._converters import _get_converter


@contextmanager
def collect_without_cyclic_gc():
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("nested", [False, True])
def test_successful_union_releases_value_and_caller(nested):
    annotation = Union[NDArray[int][:], NDArray[float][:]]
    if nested:
        annotation = Union[List[int], List[annotation]]

    @_validate_args
    def consume(value: annotation):
        pass

    def invoke():
        value = numpy.arange(32, dtype=float)
        value_ref = weakref.ref(value)
        consume([value] if nested else value)
        return value_ref

    with collect_without_cyclic_gc():
        assert invoke()() is None


def test_successful_union_conversion_releases_original_value():
    class Convertible:
        def __int__(self):
            raise ValueError("Only floating-point conversion is supported")

        def __float__(self):
            return 1.25

    with collect_without_cyclic_gc():
        value = Convertible()
        value_ref = weakref.ref(value)
        converted = _get_converter(Union[int, float])(value)
        del value
        assert converted == 1.25
        assert value_ref() is None


@pytest.mark.parametrize("factory", [_get_validator, _get_converter])
def test_rejected_union_releases_value_after_error_is_handled(factory):
    class Value:
        pass

    def invoke():
        value = Value()
        value_ref = weakref.ref(value)
        with pytest.raises(TypeError):
            factory(Union[int, float])(value)
        return value_ref

    with collect_without_cyclic_gc():
        assert invoke()() is None


def f(*args, **kwargs):
    """Bind input arguments for a function invocation."""
    return (args, kwargs)


@_validate_args
def int_func(val: int):
    assert isinstance(val, int)


@_validate_args
def union_func(val: typing.Optional[int]):
    assert isinstance(val, (int, type(None)))


@_validate_args
def anytuple_func(val: typing.Tuple):
    assert isinstance(val, tuple)


@_validate_args
def nest_tuple_func(val: typing.Tuple[typing.Tuple[int, int], "str"]):
    (i, i2), s = val

    assert isinstance(i, int)
    assert isinstance(i2, int)
    assert isinstance(s, str)


@_validate_args
def tuple_func(val: typing.Tuple[str, int]):
    s, i = val
    assert isinstance(s, str)
    assert isinstance(i, int)


@_validate_args
def ellipsis_tuple_func(val: typing.Tuple[int, ...]):
    for i in val:
        assert isinstance(i, int)


@_validate_args
def str_func(val: str):
    assert isinstance(val, str)


@_validate_args
def array_func(val: NDArray[float][..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


@_validate_args
def union_array_func(val: Union[float, NDArray[float][:]]):
    if isinstance(val, numpy.ndarray):
        assert val.ndim == 1
    else:
        assert isinstance(val, float)


@_validate_args
def tuple_array_func(
    val: Tuple[float, NDArray[float][:]],
) -> NDArray[float][:]:
    m, v = val

    assert isinstance(m, float)
    assert isinstance(v, numpy.ndarray)

    return m * v


@_validate_args
def list_int_func(val: List[int]) -> int:
    tot = 0
    for x in val:
        tot += x
    return tot


@_validate_args
def list_func(val: List) -> str:
    return ", ".join([str(x) for x in val])


@_validate_args
def list_union_func(val: List[Union[int, str]]):
    return ", ".join([str(x) for x in val])


validate_examples = [
    {
        "func": int_func,
        "valid": [f(1), f(2), f(val=2)],
        "invalid": [f(), f(1.1), f(None), f("one"), f(1, 2)],
    },
    {
        "func": str_func,
        "valid": [f("one"), f("two"), f("")],
        "invalid": [f(None), f(b"bytes"), f(1), f(1, 2)],
    },
    {
        "func": array_func,
        "valid": [
            f(numpy.array([1, 2, 3], dtype=float)),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
        ],
        "invalid": [f(None), f([[1, 2, 3]]), f(numpy.arange(30).reshape(-1, 3))],
    },
    {
        "func": union_array_func,
        "valid": [
            f(numpy.array([1, 2, 3], dtype=float)),
            f(numpy.arange(30).astype(float)),
            f(1.1),
            f(numpy.pi),
        ],
        "invalid": [
            f(None),
            f(1),
            f("1.1"),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
        ],
    },
    {
        "func": tuple_array_func,
        "valid": [f((1.1, numpy.array([1, 2, 3], dtype=float)))],
        "invalid": [
            f((1, numpy.array([1, 2, 3], dtype=float))),  # First entry type
            f((1.1, numpy.array([[1.1, 2.2], [3.3, 4.4]]))),  # Send entry shape
            f((1.1, 1.1)),
            f(None),
            f(1),
            f("1.1"),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
        ],
    },
    {
        "func": list_int_func,
        "valid": [f([1]), f([1, 2]), f([1, 2, 3])],
        "invalid": [f(1), f(1.1), f((1, 2)), f([1, "a"]), f([1, 1.1])],
    },
    {
        "func": list_func,
        "valid": [f([1]), f([1, 2]), f([1, 2, 3]), f([1, "a"]), f([1, 1.1])],
        "invalid": [f(1), f(1.1), f((1, 2)), f("a"), f(set(["a", "b"]))],
    },
    {
        "func": list_union_func,
        "valid": [f([1]), f([1, 2]), f([1, "2", 3]), f([1, "a"]), f(["a"])],
        "invalid": [f([1, 1.1]), f(1), f(1.1), f((1, 2)), f("a"), f(set(["a", "b"]))],
    },
    {
        "func": union_func,
        "valid": [f(1), f(2), f(None), f(val=2), f(val=None)],
        "invalid": [f(), f(1.1), f("one"), f(1, 2), f(None, 2)],
    },
    {
        "func": tuple_func,
        "valid": [f(("a", 1))],
        "invalid": [f((1, "a")), f(("a",)), f("a", 1), f((None, 1))],
    },
    {
        "func": nest_tuple_func,
        "valid": [f(((1, 1), "a"))],
        "invalid": [f((1, 1, "a")), f(((), "a")), f(("a",)), f("a", 1), f((None, 1))],
    },
    {
        "func": anytuple_func,
        "valid": [f(("a", 1)), f((1, "a")), f(()), f((("a", 1), 1, 1))],
        "invalid": [f("a"), f(("a", 1), 1), f(None), f()],
    },
    {
        "func": ellipsis_tuple_func,
        "valid": [f(()), f((1,)), f((1, 1)), f((1, 1, 1, 1, 1))],
        "invalid": [f(), f(1), f((1, 1, 1, "a", 1)), f(1, 1, 1)],
    },
]


@pytest.mark.parametrize("example", validate_examples)
def test_func_validation(example):
    func, valid, invalid = (example["func"], example["valid"], example["invalid"])

    for args, kwargs in valid:
        func(*args, **kwargs)

    for args, kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            func(*args, **kwargs)


@_convert_args
def int_cfunc(val: int):
    assert isinstance(val, int)


@_convert_args
def str_cfunc(val: str):
    assert isinstance(val, str)


@_convert_args
def array_cfunc(val: NDArray[float][..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


@_convert_args
def union_cfunc(val: typing.Optional[int]):
    assert isinstance(val, (int, type(None)))


@_convert_args
def tuple_cfunc(val: typing.Tuple[str, int]):
    s, i = val
    assert isinstance(s, str)
    assert isinstance(i, int)


convert_examples = [
    {
        "func": int_cfunc,
        "valid": [f(1), f(1.1), f("1"), f(2), f(val="2")],
        "invalid": [f(), f(None), f("one"), f("1.1"), f(1, 2)],
    },
    {
        "func": str_cfunc,
        "valid": [f(1), f("two"), f("bytes"), f(""), f(None), f(1.1)],
        "invalid": [f(1, 2), f()],
    },
    {
        "func": array_cfunc,
        "valid": [
            f(numpy.array([1, 2, 3], dtype=float)),
            f([[1, 2, 3]]),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
            f(numpy.arange(30).reshape(-1, 3)),
        ],
        "invalid": [f(None), f([["one", "two", "three"]])],
    },
    {
        "func": union_cfunc,
        "valid": [f(None), f(1), f(1.1), f("1")],
        "invalid": [f(), f(numpy.nan), f("one"), f(1, 2), f(None, 2)],
    },
    {
        "func": tuple_cfunc,
        "valid": [f(("a", 1))],
        "invalid": [f((1, "a")), f(("a",)), f("a", 1), f((None, 1))],
    },
]


@pytest.mark.parametrize("example", convert_examples)
def test_func_conversion(example):
    func, valid, invalid = (example["func"], example["valid"], example["invalid"])

    for args, kwargs in valid:
        func(*args, **kwargs)

    for args, kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            func(*args, **kwargs)


def test_return_annotation():
    def ret_valid(a: int, b: int) -> int:
        return a + b

    def ret_invalid(a: int, b: int) -> str:
        return a + b

    def ret_none(a: int, b: int):
        return a + b

    assert _validate_args(ret_valid)(1, 2) == 3
    assert _validate_args(ret_none)(1, 2) == 3

    with pytest.raises(TypeError):
        _validate_args(ret_invalid)(1, 2)

    assert _convert_args(ret_valid)(1, 2) == 3
    assert _convert_args(ret_none)(1, 2) == 3
    assert _convert_args(ret_invalid)(1, 2) == "3"
