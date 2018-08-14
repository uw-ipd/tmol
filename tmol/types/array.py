"""Tensor type attributes for numpy arrays."""

import enum
import attr

import typing

import numpy

from .shape import Shape

from .tensor import TensorType, _cat_internal


class Casting(enum.Enum):
    """Casting specifications for array types, see ndarray.astype."""

    no = "no"
    equiv = "equiv"
    safe = "safe"
    same_kind = "same_kind"
    unsafe = "unsafe"


@attr.s(frozen=True, auto_attribs=True, repr=False)
class NDArray(TensorType, typing._TypingBase, _root=True):
    _module: typing.ClassVar = numpy
    _tensortype: typing.ClassVar = numpy.ndarray

    dtype: numpy.dtype = attr.ib(converter=numpy.dtype)
    shape: Shape = Shape.spec[...]
    casting: Casting = attr.ib(converter=Casting, default=Casting.unsafe)

    def convert(self, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, copy=False, dtype=self.dtype)
        if not value.dtype == self.dtype:
            value = value.astype(self.dtype, casting=self.casting.value)
        if value.shape == () and [d.size for d in self.shape.dims] == [1]:
            value = value.reshape(1)

        self.validate(value)

        return value


@_cat_internal.register(numpy.ndarray)
def _cat_ndarray(first, rest, dim=0, out=None):
    return numpy.concatenate([first] + list(rest), axis=dim, out=out)
