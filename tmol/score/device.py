from typing import Optional

import torch
from .score_graph import score_graph


@score_graph
class TorchDevice:
    """Graph component specifying the target compute device.

    Attributes:
        device: The common torch compute device used for all operations.
    """

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

    device: torch.device
