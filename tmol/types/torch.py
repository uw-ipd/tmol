import attr

import typing

import numpy
import torch

from functools import singledispatch

from .shape import Shape

from .converters import get_converter
from .validators import get_validator

_numpy_torch_dtype_mapping = {
    numpy.float16: torch.float16,
    numpy.float32: torch.float32,
    numpy.float64: torch.float64,
    numpy.uint8: torch.uint8,
    numpy.int8: torch.int8,
    numpy.int16: torch.int16,
    numpy.int32: torch.int32,
    numpy.int64: torch.int64,
}


@singledispatch
def torch_dtype(dt):
    """Parse a torch dtype via numpy's dtype parsing system."""
    numeric_type = numpy.dtype(dt).type

    torch_type = _numpy_torch_dtype_mapping.get(numeric_type)
    if not torch_type:
        raise ValueError(
            f"Unsupported dtype: {dt} numeric type: {numeric_type}"
        )

    return torch_type


@torch_dtype.register(torch.dtype)
def _torch_dtype(dt):
    return dt


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Tensor(typing._TypingBase, _root=True):
    dtype: torch.dtype = attr.ib(converter=torch_dtype)
    shape: Shape = Shape.spec[...]

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )

        shape = Shape(shape)

        return attr.evolve(self, shape=shape)

    def __repr__(self):
        return f"Tensor({self.dtype!r}){self.shape!s}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def validate(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"expected {torch.Tensor!r}, received {type(value)!r}"
            )
        if not value.dtype == self.dtype:
            raise TypeError(
                f"expected {self.dtype!r}, received {value.dtype!r}"
            )
        self.shape.validate(value.shape)

        return True

    def convert(self, value):
        if isinstance(value, torch.Tensor):
            value = value
        elif isinstance(value, numpy.ndarray):
            value = torch.from_numpy(value)
        else:
            value = torch.Tensor(value)

        if not value.dtype == self.dtype:
            value = value.to(self.dtype)

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

    def _expanded_shape(self, shape):
        broadcast, *subshape = [d.size for d in self.shape.dims]
        assert broadcast is Ellipsis, (
            f"Tensor must be of broadcast shape: {self}"
        )
        assert all(
            isinstance(i, int) for i in subshape
        ), (f"Tensor must be fixed size in non-broadcast dims: {self}")

        return shape + tuple(subshape)

    def zeros(self, shape, **kwargs):
        return torch.zeros(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def ones(self, shape, **kwargs):
        return torch.ones(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def empty(self, shape, **kwargs):
        return torch.empty(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def full(self, shape, fill_value, **kwargs):
        return torch.full(
            self._expanded_shape(shape),
            fill_value,
            dtype=self.dtype,
            **kwargs,
        )


class TensorGroup:
    def __getitem__(self, idx):
        assert all(
            isinstance(a.type, Tensor) or issubclass(a.type, TensorGroup)
            for a in attr.fields(type(self))
        )

        return attr.evolve(
            self, **{
                a.name: getattr(self, a.name)[idx]
                for a in attr.fields(type(self))
            }
        )

    def __setitem__(self, idx, value):
        for a in self.__attrs_attrs__:
            getattr(self, a.name)[idx] = getattr(value, a.name)[idx]

    @classmethod
    def full(cls, shape, fill_value, **kwargs):
        return cls(
            **{
                a.name: a.type.full(shape, fill_value, **kwargs)
                for a in attr.fields(cls)
            },
        )

    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls(
            **{
                a.name: a.type.zeros(shape, **kwargs)
                for a in attr.fields(cls)
            },
        )

    @classmethod
    def ones(cls, shape, **kwargs):
        return cls(
            **{a.name: a.type.ones(shape, **kwargs)
               for a in attr.fiels(cls)},
        )

    @classmethod
    def empty(cls, shape, **kwargs):
        return cls(
            **{
                a.name: a.type.empty(shape, **kwargs)
                for a in attr.fields(cls)
            },
        )


@get_validator.register(Tensor)
def validate_tensor(tensor_type):
    return tensor_type.validate


@get_converter.register(Tensor)
def convert_tensor(tensor_type):
    return tensor_type.convert
