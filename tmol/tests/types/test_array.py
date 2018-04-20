import numpy
import pytest

from tmol.types.array import NDArray

test_examples = [
    {
        "spec": NDArray(int)[:, 3],
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
        ]
    },
    {
        "spec": NDArray(object)[:],
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
        ]
    },
]


@pytest.mark.parametrize("example", test_examples)
def test_array_validation(example):
    spec, valid, invalid = (
        example["spec"], example["valid"], example["invalid"]
    )

    for v in valid:
        assert spec.validate(v)

    for v in invalid:
        with pytest.raises((TypeError, ValueError)):
            assert not spec.validate(v)
