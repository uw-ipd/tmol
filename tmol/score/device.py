from typing import Optional

import torch
from tmol.utility.reactive import reactive_attrs

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class TorchDevice(Factory):
    @staticmethod
    def factory_for(val, device: Optional[torch.device] = None, **_):
        """Overridable clone-constructor.

        Initialize from ``val.device`` if possible, otherwise defaulting to cpu.
        """
        if device is None:
            if getattr(val, "device", None):
                device = val.device
            else:
                device = torch.device("cpu")

        return dict(device=device)

    # The target torch device
    device: torch.device = torch.device("cpu")
