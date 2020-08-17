"""Tensor type attributes for torch arrays."""

import typing
import numbers

import numpy
import torch

from functools import singledispatch

from .tensor import _TensorType, _cat_internal

_torch_dtype_mapping = {
    float: torch.float32,
    bool: torch.bool,
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
        raise ValueError(f"Unsupported dtype: {dt} numeric type: {numeric_type}")

    return torch_type


@torch_dtype.register(torch.dtype)
def _torch_dtype(dt):
    return dt


def like_kwargs(t: torch.Tensor):
    """Extract kwargs args needed to initialize an identical tensor."""
    return dict(dtype=t.dtype, layout=t.layout, device=t.device)


class Tensor(_TensorType):
    _module: typing.ClassVar = torch
    _tensortype: typing.ClassVar = torch.Tensor

    @classmethod
    def _convert_dtype(cls, t):
        return torch_dtype(t)

    @classmethod
    def convert(cls, value):
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

        if not value.dtype == cls.dtype:
            value = value.to(cls.dtype)

        if value.shape == () and [d.size for d in cls.shape.dims] == [1]:
            value = value.reshape(1)

        cls.validate(value)

        return value


@_cat_internal.register(torch.Tensor)
def _cat_tensor(first, rest, dim=0, out=None):
    return torch.cat([first] + list(rest), dim=dim, out=out)
