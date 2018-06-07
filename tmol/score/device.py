from functools import singledispatch

import torch
from tmol.utility.reactive import reactive_attrs

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class TorchDevice(Factory):
    @staticmethod
    @singledispatch
    def factory_for(other, device=None):
        """`clone`-factory, extract device from other if not provided."""
        if device is None:
            device = other.device

        return dict(device=device, )

    # The target torch device
    device: torch.device = torch.device("cpu")
