"""Utilities for numba/torch interconversion."""

import torch


def torch_cuda_array_interface(tensor):
    """__cuda_array_interface__ getter implementation for torch.Tensor."""

    if not tensor.device.type == "cuda":
        # raise AttributeError for non-cuda tensors, so that
        # hasattr(cpu_tensor, "__cuda_array_interface__") is False.
        raise AttributeError("Tensor is not on cuda device: %r" % tensor.device)

    if tensor.requires_grad:
        # RuntimeError, matching existing tensor.__array__() behavior.
        raise RuntimeError(
            "Can't get __cuda_array_interface__ on Variable that requires grad. "
            "Use var.detach().__cuda_array_interface__ instead."
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
    }[tensor.dtype]

    itemsize = tensor.storage().element_size()

    shape = tensor.shape
    strides = tuple(s * itemsize for s in tensor.stride())
    data = (tensor.data_ptr(), False)

    return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=0)


torch.Tensor.__cuda_array_interface__ = property(torch_cuda_array_interface)
