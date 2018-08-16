import numpy
import torch
import pytest

from tmol.types.torch import Tensor

validation_examples = [
    {
        "spec": Tensor(bool)[:],
        "valid": [
            torch.arange(30) > 15,
            torch.Tensor([True, True, False]).to(torch.uint8),
        ],
        "invalid": [
            # numpy types not allowed
            numpy.arange(30).reshape(10, 3),
            numpy.arange(3).reshape(1, 3),
            numpy.array([[1, 2, 3]]),
            numpy.array([[1.0, 2.0, 3.0]]).astype(int),
            numpy.array([[0, 1, 1]]).astype("u1"),
            # bad shape
            numpy.arange(30).reshape(3, 10),
            torch.arange(30, dtype=torch.int64).reshape(3, 10),
            numpy.arange(3),
            # defaults to floating-point types
            torch.Tensor([True, True, False]),
            torch.Tensor([[1, 2, 3]]),
            torch.arange(30),
            # no casting
            numpy.arange(30).reshape(10, 3).astype(float),
            numpy.array([[1.0, 2.0, 3.0]]),
            [1, 2, 3],
            [[1, 2, 3]],
            numpy.array([["one", "two", "three"]]),
        ],
    },
    {
        "spec": Tensor(int)[:, 3],
        "valid": [
            torch.arange(30, dtype=torch.int64).reshape(10, 3),
            torch.Tensor([[1, 2, 3]]).to(torch.int64),
        ],
        "invalid": [
            # numpy types not allowed
            numpy.arange(30).reshape(10, 3),
            numpy.arange(3).reshape(1, 3),
            numpy.array([[1, 2, 3]]),
            numpy.array([[1.0, 2.0, 3.0]]).astype(int),
            # bad shape
            numpy.arange(30).reshape(3, 10),
            torch.arange(30, dtype=torch.int64).reshape(3, 10),
            numpy.arange(3),
            # defaults to floating-point types
            torch.Tensor([[1, 2, 3]]),
            torch.arange(30),
            # no casting
            numpy.arange(30).reshape(10, 3).astype(float),
            numpy.array([[1.0, 2.0, 3.0]]),
            [1, 2, 3],
            [[1, 2, 3]],
            numpy.array([["one", "two", "three"]]),
        ],
    },
    {
        "spec": Tensor("f")[:],
        "valid": [torch.arange(30), torch.Tensor([1, 2, 3])],
        "invalid": [
            # bad shape
            torch.arange(30).reshape(3, 10),
            torch.arange(30).to(torch.int32),
            # no float-float casting
            numpy.array([1.0, 2.0, 3.0]).astype("f"),
            numpy.arange(30).astype("f"),
            numpy.arange(30),
            numpy.arange(30, dtype=float),
            numpy.array([1.0, 2.0, 3.0]),
            numpy.arange(30).reshape(10, 3).astype("f"),
            numpy.array([[1.0, 2.0, 3.0]]),
            [1, 2, 3],
            [[1, 2, 3]],
            numpy.array([["one", "two", "three"]]),
        ],
    },
    {
        "spec": Tensor(float)[:],
        "valid": [torch.arange(30), torch.Tensor([1, 2, 3])],
        "invalid": [
            # bad shape
            torch.arange(30).reshape(3, 10),
            torch.arange(30).to(torch.float64),
            # no float-float casting
            numpy.array([1.0, 2.0, 3.0]).astype("f"),
            numpy.arange(30).astype("f"),
            numpy.arange(30),
            numpy.arange(30, dtype=float),
            numpy.array([1.0, 2.0, 3.0]),
            numpy.arange(30).reshape(10, 3).astype("f"),
            numpy.array([[1.0, 2.0, 3.0]]),
            [1, 2, 3],
            [[1, 2, 3]],
            numpy.array([["one", "two", "three"]]),
        ],
    },
]


@pytest.mark.parametrize("example", validation_examples)
def test_array_validation(example):
    spec, valid, invalid = (example["spec"], example["valid"], example["invalid"])

    for v in valid:
        assert spec.validate(v)
        assert isinstance(v, spec)

    for v in invalid:
        assert not isinstance(v, spec)
        with pytest.raises((TypeError, ValueError)):
            assert not spec.validate(v)


invalid_dtypes = [numpy.dtype([("coord", float, 3), ("val", int)]), numpy.complex, "c8"]


@pytest.mark.parametrize("invalid_dtype", invalid_dtypes)
def test_invalid_dtype(invalid_dtype):
    with pytest.raises(ValueError):
        Tensor(invalid_dtype)


converstion_examples = [
    {
        "spec": Tensor(float)[3],
        "conversions": [
            ([1, 2, 3], torch.Tensor([1, 2, 3])),
            ([True, True, False], torch.Tensor([1, 1, 0])),
            (torch.arange(3) < 2, torch.Tensor([1, 1, 0])),
            ([numpy.pi] * 3, torch.Tensor([numpy.pi] * 3)),
        ],
        "invalid": [
            # Invalid casts.
            numpy.array(list("abc")),
            numpy.array(["a", "b", "c"], dtype=object),
            ["one", "two", "three"],
            # No shape coercion
            numpy.arange(30).reshape(10, 3),
            numpy.arange(3).reshape(1, 3),
            [[1, 2, 3]],
        ],
    },
    {
        "spec": Tensor(float)[1],
        "conversions": [
            (numpy.pi, torch.Tensor([numpy.pi])),
            (1663, torch.Tensor([1663])),
            ([1663], torch.Tensor([1663])),
            (numpy.arange(10)[5], torch.Tensor([5])),
            (torch.arange(10)[5], torch.Tensor([5])),
            (numpy.arange(10)[6:7], torch.Tensor([6])),
            (torch.arange(10)[6:7], torch.Tensor([6])),
        ],
        "invalid": [numpy.array(["one"]), torch.arange(3)],
    },
    {
        "spec": Tensor(int)[1],
        "conversions": [
            (numpy.pi, torch.Tensor([3]).to(torch.long)),
            ([1], torch.Tensor([1]).to(torch.long)),
        ],
        "invalid": [
            numpy.array(["one"]),
            torch.arange(3),
            torch.arange(3).to(torch.long),
        ],
    },
    {
        "spec": Tensor(bool)[:],
        "conversions": [
            ([True, True, True], torch.Tensor([1, 1, 1]).to(torch.uint8)),
            (numpy.arange(3) < 2, torch.Tensor([1, 1, 0]).to(torch.uint8)),
            (torch.arange(3), torch.Tensor([0, 1, 2]).to(torch.uint8)),
        ],
        "invalid": [numpy.array(["one"])],
    },
]


@pytest.mark.parametrize("example", converstion_examples)
def test_array_conversion(example):
    spec, conversions, invalid = (
        example["spec"],
        example["conversions"],
        example["invalid"],
    )

    for f, t in conversions:
        res = spec.convert(f)
        numpy.testing.assert_array_equal(res, t)
        assert res.dtype == t.dtype

    for v in invalid:
        with pytest.raises((TypeError, ValueError)):
            assert not spec.convert(v)
