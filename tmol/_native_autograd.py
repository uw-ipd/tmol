"""Keep autograd ownership and saved-tensor checks on PyTorch's Python API."""

import torch


def _bind(forward, backward):
    class NativeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, arguments, gradient_indices, *args):
            ctx._tmol_gradient_indices = gradient_indices
            ctx._tmol_saved_data = {}
            result = forward(ctx, arguments)
            return tuple(result) if isinstance(result, list) else result

        @staticmethod
        def backward(ctx, *grads):
            result = backward(ctx, grads)
            return (None, None, *(result[i] for i in ctx._tmol_gradient_indices))

    return NativeFunction.apply
