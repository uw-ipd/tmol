import torch
from tmol.utility.reactive import reactive_attrs


@reactive_attrs(auto_attribs=True)
class TorchDevice:
    # The target torch device
    device: torch.device = torch.device("cpu")
