"""Keep autograd ownership and saved-tensor checks on PyTorch's Python API."""

import torch


def _bind(forward, backward):
    return type(
        "NativeFunction",
        (torch.autograd.Function,),
        {"forward": staticmethod(forward), "backward": staticmethod(backward)},
    ).apply
