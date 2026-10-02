import torch


def resolve_device(device: torch.device) -> torch.device:
    """Resolve an unindexed CUDA device to the current device and any CPU to "cpu"."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu") if device.type == "cpu" else device


def synchronize_device(device: torch.device) -> None:
    """Wait for queued CUDA work on ``device``; do nothing for CPU devices."""
    device = resolve_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
