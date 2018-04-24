import enum
import attr

import typing

import numpy

from .shape import Shape

from .converters import get_converter
from .validators import get_validator


class Casting(enum.Enum):
    """Casting specifications for array types, see ndarray.astype."""
    no = "no"
    equiv = "equiv"
    safe = "safe"
    same_kind = "same_kind"
    unsafe = "unsafe"


@attr.s(frozen=True, auto_attribs=True, repr=False)
class NDArray(typing._TypingBase, _root=True):
    dtype: numpy.dtype = attr.ib(converter=numpy.dtype)
    shape: Shape = Shape.spec[...]
    casting: Casting = attr.ib(converter=Casting, default=Casting.unsafe)

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )

        shape = Shape(shape)

        return attr.evolve(self, shape=shape)

    def __repr__(self):
        return f"NDArray({self.dtype!r}){self.shape!s}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def validate(self, value):
        if not isinstance(value, numpy.ndarray):
            raise TypeError(
                f"expected {numpy.ndarray!r}, received {type(value)!r}"
            )
        if not value.dtype == self.dtype:
            raise TypeError(
                f"expected {self.dtype!r}, received {value.dtype!r}"
            )
        self.shape.validate(value.shape)

        return True

    def convert(self, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value, copy=False, dtype=self.dtype)
        if not value.dtype == self.dtype:
            value = value.astype(self.dtype, casting=self.casting.value)
        if value.shape == () and [d.size for d in self.shape.dims] == [1]:
            value = value.reshape(1)

        self.validate(value)

        return value

    def __instancecheck__(self, obj):
        """Overloaded isinstance to check type and shape."""

        try:
            self.validate(obj)
            return True
        except (TypeError, ValueError):
            return False


@get_validator.register(NDArray)
def validate_ndarray(ndarray_type):
    return ndarray_type.validate


@get_converter.register(NDArray)
def convert_ndarray(ndarray_type):
    return ndarray_type.convert
