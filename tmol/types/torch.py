import attr

import typing
import numbers

import numpy
import torch

from functools import singledispatch

from .shape import Shape

from .tensor import TensorType

_torch_dtype_mapping = {
    float: torch.float32,
    bool: torch.uint8,
    numpy.float16: torch.float16,
    numpy.float32: torch.float32,
    numpy.float64: torch.float64,
    numpy.uint8: torch.uint8,
    numpy.bool_: torch.uint8,
    numpy.int8: torch.int8,
    numpy.int16: torch.int16,
    numpy.int32: torch.int32,
    numpy.int64: torch.int64,
}


@singledispatch
def torch_dtype(dt):
    """Resolve a torch dtype via numpy's dtype parsing system."""
    explict_dtype = _torch_dtype_mapping.get(dt, None)

    if explict_dtype:
        return explict_dtype

    numeric_type = numpy.dtype(dt).type

    torch_type = _torch_dtype_mapping.get(numeric_type, None)
    if not torch_type:
        raise ValueError(
            f"Unsupported dtype: {dt} numeric type: {numeric_type}"
        )

    return torch_type


@torch_dtype.register(torch.dtype)
def _torch_dtype(dt):
    return dt


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Tensor(TensorType, typing._TypingBase, _root=True):
    _module: typing.ClassVar = torch
    _tensortype: typing.ClassVar = torch.Tensor

    dtype: torch.dtype = attr.ib(converter=torch_dtype)
    shape: Shape = Shape.spec[...]

    def convert(self, value):
        if isinstance(value, torch.Tensor):
            value = value
        elif isinstance(value, numpy.ndarray):
            if value.dtype.type == numpy.bool_:
                value = value.astype("u1")
            value = torch.from_numpy(value)
        elif isinstance(value, numbers.Number):
            value = torch.Tensor([value])[0]
        else:
            value = torch.Tensor(value)

        if not value.dtype == self.dtype:
            value = value.to(self.dtype)

        if value.shape == () and [d.size for d in self.shape.dims] == [1]:
            value = value.reshape(1)

        self.validate(value)

        return value
