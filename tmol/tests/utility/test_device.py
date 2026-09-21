import pytest
import torch

from tmol.utility._device import synchronize_device


def test_synchronize_device_is_a_cpu_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_synchronize(device: torch.device) -> None:
        raise AssertionError(f"unexpected CUDA synchronization on {device}")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_synchronize)

    synchronize_device(torch.device("cpu"))


def test_synchronize_device_targets_a_concrete_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    synchronize_device(torch.device("cuda"))

    assert synchronized == [torch.device("cuda", 2)]
