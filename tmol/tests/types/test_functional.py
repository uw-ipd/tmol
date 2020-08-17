import numpy
import pytest

import typing
from typing import Union, Tuple

from tmol.types.functional import validate_args, convert_args
from tmol.types.array import NDArray


def f(*args, **kwargs):
    """Bind input arguments for a function invocation."""
    return (args, kwargs)


@validate_args
def int_func(val: int):
    assert isinstance(val, int)


@validate_args
def union_func(val: typing.Optional[int]):
    assert isinstance(val, (int, type(None)))


@validate_args
def anytuple_func(val: typing.Tuple):
    assert isinstance(val, tuple)


@validate_args
def nest_tuple_func(val: typing.Tuple[typing.Tuple[int, int], "str"]):
    (i, i2), s = val

    assert isinstance(i, int)
    assert isinstance(i2, int)
    assert isinstance(s, str)


@validate_args
def tuple_func(val: typing.Tuple[str, int]):
    s, i = val
    assert isinstance(s, str)
    assert isinstance(i, int)


@validate_args
def ellipsis_tuple_func(val: typing.Tuple[int, ...]):
    for i in val:
        assert isinstance(i, int)


@validate_args
def str_func(val: str):
    assert isinstance(val, str)


@validate_args
def array_func(val: NDArray[float][..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


@validate_args
def union_array_func(val: Union[float, NDArray[float][:]]):
    if isinstance(val, numpy.ndarray):
        assert val.ndim == 1
    else:
        assert isinstance(val, float)


@validate_args
def tuple_array_func(val: Tuple[float, NDArray[float][:]],) -> NDArray[float][:]:
    m, v = val

    assert isinstance(m, float)
    assert isinstance(v, numpy.ndarray)

    return m * v


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


@convert_args
def int_cfunc(val: int):
    assert isinstance(val, int)


@convert_args
def str_cfunc(val: str):
    assert isinstance(val, str)


@convert_args
def array_cfunc(val: NDArray[float][..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


@convert_args
def union_cfunc(val: typing.Optional[int]):
    assert isinstance(val, (int, type(None)))


@convert_args
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

    assert validate_args(ret_valid)(1, 2) == 3
    assert validate_args(ret_none)(1, 2) == 3

    with pytest.raises(TypeError):
        validate_args(ret_invalid)(1, 2)

    assert convert_args(ret_valid)(1, 2) == 3
    assert convert_args(ret_none)(1, 2) == 3
    assert convert_args(ret_invalid)(1, 2) == "3"
