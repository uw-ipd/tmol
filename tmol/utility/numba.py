"""Utilities for numba/torch interconversion.

The primary functions provided "to_cuda_array" and "is_cuda_array", backports of
numba 0.39 memory management utility functions, updated with single-dispatch based
adaptors for torch tensors.
"""

import attr
from functools import singledispatch

import torch

from ctypes import c_ulong
import numpy as np
import numba
from numba.cuda.cudadrv import devicearray, devices, driver
import numba.cuda.simulator as simulator

assert (
    numba.__version__ < "0.39"
), "Backport of numba v0.39 device array interface no longer needed."

try:
    long
except NameError:
    long = int

require_context = devices.require_context
current_context = devices.get_context


def _get_devptr_for_active_ctx(ptr):
    """Hack of internal numba 0.39 api, always assume pointer is on device."""
    return c_ulong(ptr)


@require_context
def from_cuda_array_interface(desc, owner=None):
    """Create a DeviceNDArray from a cuda-array-interface description.
    The *owner* is the owner of the underlying memory.
    The resulting DeviceNDArray will acquire a reference from it.
    """
    shape = desc["shape"]
    strides = desc.get("strides")
    dtype = np.dtype(desc["typestr"])

    shape, strides, dtype = _prepare_shape_strides_dtype(
        shape, strides, dtype, order="C"
    )

    devptr = _get_devptr_for_active_ctx(desc["data"][0])
    data = driver.MemoryPointer(
        current_context(), devptr, size=np.prod(shape) * dtype.itemsize, owner=owner
    )
    da = devicearray.DeviceNDArray(
        shape=shape, strides=strides, dtype=dtype, gpu_data=data
    )
    return da


@singledispatch
def as_cuda_array(obj):
    """Create a DeviceNDArray from any object that implements the
    cuda-array-interface.

    A view of the underlying GPU buffer is created.  No copying of the data is
    done.  The resulting DeviceNDArray will acquire a reference from ``obj``.

    Shim interface supports creation of *fake* cuda arrays iff the cuda
    simulator is active as the current context.
    """

    if isinstance(
        numba.cuda.current_context(), simulator.cudadrv.devices.FakeCUDAContext
    ):
        if is_cuda_array(obj):
            raise ValueError("Can not create simulator device array from cuda array.")
        return simulator.cudadrv.devicearray.FakeCUDAArray(obj.__array__())

    if not is_cuda_array(obj):
        raise TypeError("*obj* doesn't implement the cuda array interface.")
    else:
        return from_cuda_array_interface(obj.__cuda_array_interface__, owner=obj)


@singledispatch
def is_cuda_array(obj):
    """Test if the object has defined the ``__cuda_array_interface__``.
    Does not verify the validity of the interface.
    """
    return hasattr(obj, "__cuda_array_interface__")


def _prepare_shape_strides_dtype(shape, strides, dtype, order):
    dtype = np.dtype(dtype)
    if isinstance(shape, (int, long)):
        shape = (shape,)
    if isinstance(strides, (int, long)):
        strides = (strides,)
    else:
        if shape == ():
            shape = (1,)
        strides = strides or _fill_stride_by_order(shape, dtype, order)
    return shape, strides, dtype


def _fill_stride_by_order(shape, dtype, order):
    nd = len(shape)
    strides = [0] * nd
    if order == "C":
        strides[-1] = dtype.itemsize
        for d in reversed(range(nd - 1)):
            strides[d] = strides[d + 1] * shape[d + 1]
    elif order == "F":
        strides[0] = dtype.itemsize
        for d in range(1, nd):
            strides[d] = strides[d - 1] * shape[d - 1]
    else:
        raise ValueError("must be either C/F order")
    return tuple(strides)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class TorchCUDAArrayAdaptor:
    t: torch.Tensor

    @property
    def __cuda_array_interface__(self):
        if not self.t.device.type == "cuda":
            raise ValueError(f"tensor is not on cuda device: {self.t.device !r}")
        if not self.t.device.index == numba.cuda.get_current_device().id:
            raise ValueError(
                f"tensor device: {self.t !r} "
                f"is not active numba context: {numba.cuda.current_context()!r}"
            )

        typestr = {
            torch.float16: "f2",
            torch.float32: "f4",
            torch.float64: "f8",
            torch.uint8: "u1",
            torch.int8: "i1",
            torch.int16: "i2",
            torch.int32: "i4",
            torch.int64: "i8",
        }[self.t.dtype]

        itemsize = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.uint8: 1,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
        }[self.t.dtype]

        shape = self.t.shape
        strides = tuple(s * itemsize for s in self.t.stride())
        data = (self.t.data_ptr(), False)

        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=0)

    @staticmethod
    @as_cuda_array.register(torch.Tensor)
    def as_cuda_array(torch_tensor):
        if isinstance(
            numba.cuda.current_context(), simulator.cudadrv.devices.FakeCUDAContext
        ):
            # Fall through to default implementation iff in simulator context
            return as_cuda_array.dispatch(object)(torch_tensor)

        return as_cuda_array(TorchCUDAArrayAdaptor(torch_tensor))

    @staticmethod
    @is_cuda_array.register(torch.Tensor)
    def is_cuda_array(torch_tensor):
        return torch_tensor.device.type == "cuda"
