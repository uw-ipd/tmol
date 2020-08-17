import numpy
import pytest

from tmol.types.array import NDArray

validation_examples = [
    {
        "spec": NDArray[int][:, 3],
        "valid": [
            numpy.arange(30).reshape(10, 3),
            numpy.arange(3).reshape(1, 3),
            numpy.array([[1, 2, 3]]),
            numpy.array([[1.0, 2.0, 3.0]]).astype(int),
        ],
        "invalid": [
            # bad shape
            numpy.arange(30).reshape(3, 10),
            numpy.arange(3),
            # no casting
            numpy.arange(30).reshape(10, 3).astype(float),
            numpy.array([[1.0, 2.0, 3.0]]),
            [1, 2, 3],
            [[1, 2, 3]],
            numpy.array([["one", "two", "three"]]),
        ],
    },
    {
        "spec": NDArray[object][:],
        "valid": [
            numpy.array(["one", "two", "three"], dtype=object),
            numpy.array([1, 2, 3], dtype=object),
        ],
        "invalid": [
            numpy.array(["one", "two", "three"]),  # string dtype
            numpy.arange(30).reshape(10, 3),
            numpy.arange(3).reshape(1, 3),
            numpy.arange(30),
            numpy.arange(30).reshape(3, 10),
            numpy.arange(3),
            [1, 2, 3],
            [[1, 2, 3]],
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


packed_dtype = numpy.dtype([("coord", float, 3), ("val", int)])
incompatible_dtype = numpy.dtype([("coord", float, 3), ("val", float)])

converstion_examples = [
    {
        "spec": NDArray[packed_dtype][:],
        "conversions": [
            (
                [((1, 1, 1), 10), ((2, 2, 2), 20)],
                numpy.array(
                    [((1.0, 1.0, 1.0), 10), ((2.0, 2.0, 2.0), 20)], dtype=packed_dtype
                ),
            ),
            (numpy.zeros(3, dtype=packed_dtype), numpy.zeros(3, packed_dtype)),
        ],
        "invalid": [
            numpy.zeros(3, dtype=incompatible_dtype),
            numpy.zeros((1, 10), dtype=packed_dtype),
            numpy.arange(4),
        ],
    },
    {
        "spec": NDArray[int][3],
        "conversions": [
            ([1, 2, 3], numpy.array([1, 2, 3], dtype=int)),
            ([True, True, False], numpy.array([1, 1, 0], dtype=int)),
            ([1.1, 2.2, 3.3], numpy.array([1, 2, 3], dtype=int)),
            (numpy.linspace(1.1, 1.5, 3), numpy.array([1, 1, 1], dtype=int)),
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
        "spec": NDArray[float][1],
        "conversions": [
            (numpy.pi, numpy.array([numpy.pi])),
            (1663, numpy.array([1663], dtype=float)),
            ([1663], numpy.array([1663], dtype=float)),
            (numpy.arange(10)[5], numpy.array(5, dtype=float)),
            (numpy.arange(10)[6:7], numpy.array(6, dtype=float)),
        ],
        "invalid": [
            numpy.array(["one"]),
            numpy.arange(3),
            numpy.arange(3).astype(float),
        ],
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
