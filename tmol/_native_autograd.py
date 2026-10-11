"""Keep autograd ownership and saved-tensor checks on PyTorch's Python API."""

import torch

_functions = {}


def _bind(forward, backward):
    class NativeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, arguments, *args):
            ctx._tmol_saved_data = {}
            result = forward(ctx, arguments)
            return tuple(result) if isinstance(result, list) else result

        @staticmethod
        def backward(ctx, *grads):
            return (None, *backward(ctx, grads))

    return NativeFunction.apply
