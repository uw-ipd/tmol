"""Tensor type attributes for numpy arrays."""

import enum

import typing

import numpy

from .tensor import _TensorType, _cat_internal


class Casting(enum.Enum):
    """Casting specifications for array types, see ndarray.astype."""

    no = "no"
    equiv = "equiv"
    safe = "safe"
    same_kind = "same_kind"
    unsafe = "unsafe"


class NDArray(_TensorType):
    _module: typing.ClassVar = numpy
    _tensortype: typing.ClassVar = numpy.ndarray

    casting: Casting = Casting.unsafe

    @classmethod
    def _convert_dtype(cls, t):
        return numpy.dtype(t)

    @classmethod
    def convert(cls, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, copy=False, dtype=cls.dtype)
        if not value.dtype == cls.dtype:
            value = value.astype(cls.dtype, casting=cls.casting.value)
        if value.shape == () and [d.size for d in cls.shape.dims] == [1]:
            value = value.reshape(1)

        cls.validate(value)

        return value


@_cat_internal.register(numpy.ndarray)
def _cat_ndarray(first, rest, dim=0, out=None):
    return numpy.concatenate([first] + list(rest), axis=dim, out=out)
