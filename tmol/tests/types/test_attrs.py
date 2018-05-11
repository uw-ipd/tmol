import pytest
import attr
from tmol.types.attrs import ValidateAttrs, ConvertAttrs


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ValidateObj(ValidateAttrs):
    a: int
    b: str


def f(*args, **kwargs):
    """Bind input arguments for a function invocation."""
    return (args, kwargs)


validate_examples = [
    {
        "func": ValidateObj,
        "expected": {
            "a": 1,
            "b": "b"
        },
        "valid": [
            f(a=1, b="b"),
            f(1, "b"),
        ],
        "invalid": [
            f(1, 1),
            f(1),
            f("b", 1),
            f((1, "b")),
        ]
    },
]


@pytest.mark.parametrize("example", validate_examples)
def test_validate_attrs(example):
    func, expected, valid, invalid = (
        example["func"],
        example["expected"],
        example["valid"],
        example["invalid"],
    )

    for args, kwargs in valid:
        res = func(*args, **kwargs)
        assert attr.asdict(res) == expected

    for args, kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            func(*args, **kwargs)


@pytest.mark.xfail
def test_set_post_init():
    """Object validation does not function on setattr, only on init."""
    v = ValidateObj(1, "abc")
    with pytest.raises(TypeError):
        v.a = "one"


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ConvertObj(ConvertAttrs):
    a: int
    b: int


convert_examples = [
    {
        "func": ConvertObj,
        "expected": {
            "a": 1,
            "b": 1
        },
        "valid": [
            f(a=1, b="1"),
            f(1, 1),
        ],
        "invalid": [
            f(1),
            f("b", 1),
            f((1, "b")),
        ]
    },
]


@pytest.mark.parametrize("example", convert_examples)
def test_convert_attrs(example):
    func, expected, valid, invalid = (
        example["func"],
        example["expected"],
        example["valid"],
        example["invalid"],
    )

    for args, kwargs in valid:
        res = func(*args, **kwargs)
        assert attr.asdict(res) == expected

    for args, kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            func(*args, **kwargs)
