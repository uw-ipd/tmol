import gc
import weakref

import pytest
import torch

from tmol.score.common._cuda_graph import CapturedScoringGraph
from tmol.tests import requires_cuda

pytestmark = [
    requires_cuda,
    pytest.mark.filterwarnings(
        "ignore:The AccumulateGrad node's stream does not match"
    ),
]


@pytest.fixture
def torch_device():
    return torch.device("cuda", torch.cuda.current_device())


class _ScoringExample(torch.nn.Module):
    def __init__(self, device, dtype, trainable=True, unused_coords=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(1, 6, device=device, dtype=dtype) / 4,
            requires_grad=trainable,
        )
        self.register_buffer("offset", torch.ones(5, device=device, dtype=dtype) / 2)
        self.unused_coords = unused_coords

    def forward(self, coords):
        if self.unused_coords:
            return self.weight.unsqueeze(0).expand(3, -1) + self.offset
        if self.training:
            return coords.square() * self.weight + self.offset
        return coords * self.weight - self.offset


@pytest.mark.parametrize("trainable,layout", [(False, "strided"), (True, "negative")])
def test_cuda_scoring_capture_replays_values_and_parameter_gradients(
    torch_device, trainable, layout
):
    dtype = torch.float32
    module = _ScoringExample(torch_device, dtype, trainable)
    sample = torch.zeros((3, 5), device=torch_device, dtype=dtype, requires_grad=True)
    capture = CapturedScoringGraph(module, sample)
    for shift in (0, -2):
        coords = (
            torch.arange(15, device=torch_device, dtype=dtype).reshape(3, 5) + shift
        ) / 8
        if layout == "strided":
            coords = coords.T.contiguous().T
        elif layout == "negative":
            coords = torch._neg_view(coords)
        coords = coords.detach().requires_grad_(True)
        actual = capture(coords).clone()
        expected = coords.square() * module.weight + module.offset
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        inputs = (coords, module.weight) if trainable else (coords,)
        actual_grads = torch.autograd.grad(actual.sum(), inputs)
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    capture.eval()
    torch.testing.assert_close(
        capture(coords), coords * module.weight - module.offset, atol=0, rtol=0
    )


def test_cuda_scoring_capture_preserves_unused_coordinate_gradients(torch_device):
    module = _ScoringExample(torch_device, torch.float32, unused_coords=True)
    sample = torch.zeros((3, 5), device=torch_device, requires_grad=True)
    capture = CapturedScoringGraph(module, sample)
    coords = torch.ones_like(sample, requires_grad=True)
    coord_grad, weight_grad = torch.autograd.grad(
        capture(coords).sum(), (coords, module.weight), allow_unused=True
    )
    assert coord_grad is None
    torch.testing.assert_close(
        weight_grad, torch.full_like(module.weight, 3), atol=0, rtol=0
    )


def test_cuda_scoring_capture_lives_until_pending_backward_finishes(torch_device):
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        module = _ScoringExample(torch_device, torch.float64)
        sample = torch.zeros(
            (3, 5), device=torch_device, dtype=torch.float64, requires_grad=True
        )
        capture = CapturedScoringGraph(module, sample)
        capture_ref = weakref.ref(capture)
        module_ref = weakref.ref(module)
        coords = torch.ones_like(sample, requires_grad=True)
        output = capture(coords).clone()
        weight = module.weight
        del capture, module, sample
        assert capture_ref() is not None
        assert module_ref() is not None
        coord_grad, weight_grad = torch.autograd.grad(output.sum(), (coords, weight))
        del output
        assert capture_ref() is None
        assert module_ref() is None
        # Returned gradients own their storage even after the graphs die.
        torch.testing.assert_close(
            coord_grad, 2 * weight.expand_as(coords), atol=0, rtol=0
        )
        torch.testing.assert_close(
            weight_grad, torch.full_like(weight, 3), atol=0, rtol=0
        )
    finally:
        if gc_enabled:
            gc.enable()


@pytest.mark.parametrize("device_index", [0, 1])
def test_cuda_scoring_capture_preserves_device_and_caller_stream(device_index):
    if torch.cuda.device_count() <= device_index:
        pytest.skip("Requested CUDA device is unavailable")
    original_device = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    module = _ScoringExample(device, torch.float32)
    sample = torch.zeros((3, 5), device=device, requires_grad=True)
    capture = CapturedScoringGraph(module, sample)
    assert torch.cuda.current_device() == original_device
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        coords = torch.full_like(sample, 2, requires_grad=True)
        score = capture(coords)
        (grad,) = torch.autograd.grad(score.sum(), coords)
    stream.synchronize()
    torch.testing.assert_close(
        score, 4 * module.weight.expand_as(coords) + module.offset, atol=0, rtol=0
    )
    torch.testing.assert_close(
        grad, 4 * module.weight.expand_as(coords), atol=0, rtol=0
    )
    assert torch.cuda.current_device() == original_device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_scoring_capture_preserves_uncached_autocast(torch_device, dtype):
    module = torch.nn.Linear(5, 3, bias=False, device=torch_device)
    with torch.no_grad():
        module.weight.copy_(torch.arange(15, device=torch_device).reshape(3, 5) / 8)
    sample = torch.zeros((4, 5), device=torch_device, requires_grad=True)
    with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
        capture = CapturedScoringGraph(module, sample)
    for shift in (0, -2):
        coords = (
            (torch.arange(20, device=torch_device).reshape(4, 5) + shift) / 8
        ).requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            actual = capture(coords).clone()
            expected = torch.nn.functional.linear(coords, module.weight)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual_grads = torch.autograd.grad(actual.sum(), (coords, module.weight))
        expected_grads = torch.autograd.grad(expected.sum(), (coords, module.weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
