import torch


def resolve_device(device: torch.device) -> torch.device:
    """Make device concrete; assigns user-provided device('cuda') a device number"""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device
