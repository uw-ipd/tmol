import numpy
import pytest

from tmol.types.functional import validate_args, convert_args
from tmol.types.array import NDArray


def f(*args, **kwargs):
    """Bind input arguments for a function invocation."""
    return (args, kwargs)


@validate_args
def int_func(val: int):
    assert isinstance(val, int)


@validate_args
def str_func(val: str):
    assert isinstance(val, str)


@validate_args
def array_func(val: NDArray(float)[..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


validate_examples = [
    {
        "func": int_func,
        "valid": [
            f(1),
            f(2),
            f(val=2),
        ],
        "invalid": [
            f(),
            f(1.1),
            f(None),
            f("one"),
            f(1, 2),
        ]
    },
    {
        "func": str_func,
        "valid": [
            f("one"),
            f("two"),
            f(""),
        ],
        "invalid": [
            f(None),
            f(b"bytes"),
            f(1),
            f(1, 2),
        ]
    },
    {
        "func": array_func,
        "valid": [
            f(numpy.array([1, 2, 3], dtype=float)),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
        ],
        "invalid": [
            f(None),
            f([[1, 2, 3]]),
            f(numpy.arange(30).reshape(-1, 3)),
        ]
    },
]


@pytest.mark.parametrize("example", validate_examples)
def test_func_validation(example):
    func, valid, invalid = (
        example["func"], example["valid"], example["invalid"]
    )

    for args, kwargs in valid:
        try:
            func(*args, **kwargs)
        except (TypeError, ValueError) as ex:
            assert not ex, f"Validation error. func: {func} args: {args} kwargs: {kwargs}"

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
def array_cfunc(val: NDArray(float)[..., 3]):
    assert isinstance(val, numpy.ndarray)
    assert val.ndim >= 1
    assert val.shape[-1] == 3


convert_examples = [
    {
        "func": int_cfunc,
        "valid": [
            f(1),
            f(1.1),
            f("1"),
            f(2),
            f(val="2"),
        ],
        "invalid": [
            f(),
            f(None),
            f("one"),
            f("1.1"),
            f(1, 2),
        ]
    },
    {
        "func": str_cfunc,
        "valid": [
            f(1),
            f("two"),
            f("bytes"),
            f(""),
            f(None),
            f(1.1),
        ],
        "invalid": [
            f(1, 2),
            f(),
        ]
    },
    {
        "func": array_cfunc,
        "valid": [
            f(numpy.array([1, 2, 3], dtype=float)),
            f([[1, 2, 3]]),
            f(numpy.arange(30).reshape(-1, 3).astype(float)),
            f(numpy.arange(30).reshape(-1, 3)),
        ],
        "invalid": [f(None), f([["one", "two", "three"]])]
    },
]


@pytest.mark.parametrize("example", convert_examples)
def test_func_conversion(example):
    func, valid, invalid = (
        example["func"], example["valid"], example["invalid"]
    )

    for args, kwargs in valid:
        try:
            func(*args, **kwargs)
        except (TypeError, ValueError) as ex:
            assert not ex, f"Validation error. func: {func} args: {args} kwargs: {kwargs}"

    for args, kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            func(*args, **kwargs)
