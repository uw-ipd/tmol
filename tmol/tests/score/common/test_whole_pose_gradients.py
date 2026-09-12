import pytest
import torch


@pytest.fixture(scope="module")
def gradient_module():
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        [
            "whole_pose_gradients.pybind.cpp",
            "../../../score/common/whole_pose_scoring.cuda.cu",
        ],
        "tmol.tests.score.common._whole_pose_gradients",
    )


def reference(saved, weights):
    channels, atoms, xyz = saved.shape
    poses = weights.shape[1]
    weighted = saved.reshape(channels, poses, atoms // poses, xyz) * weights.reshape(
        channels, poses, 1, 1
    )
    return (weighted[0] if channels == 1 else weighted.sum(0)).reshape(atoms, xyz)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "layout",
    [
        "contiguous",
        "weight_stride",
        "broadcast",
        "saved_stride",
        "negative_weight",
        "negative_saved",
    ],
)
def test_weighted_gradient_layouts(
    gradient_module, torch_device, dtype, channels, layout
):
    generator = torch.Generator(device=torch_device).manual_seed(20260911)
    saved = (
        torch.randn(
            channels, 17 * 103, 3, device=torch_device, dtype=dtype, generator=generator
        )
        * 1000
    )
    weights = torch.randn(
        channels, 17, device=torch_device, dtype=dtype, generator=generator
    )
    if layout == "weight_stride":
        backing = torch.empty((17, channels * 2 + 1), device=torch_device, dtype=dtype)
        view = backing[:, 1::2].t()
        view.copy_(weights)
        weights = view
    elif layout == "broadcast":
        weights = weights[:1, :1].expand(channels, 17)
    elif layout == "saved_stride":
        backing = torch.empty((*saved.shape[:2], 7), device=torch_device, dtype=dtype)
        view = backing[..., 1::2]
        view.copy_(saved)
        saved = view
    if layout == "negative_weight":
        weights = torch._neg_view(weights)
    elif layout == "negative_saved":
        saved = torch._neg_view(saved)
    with torch.no_grad():
        expected = reference(saved, weights)
        actual = gradient_module.accumulate(saved, weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_weighted_gradient_preserves_derivative_graph(gradient_module, torch_device):
    saved = torch.randn(
        3, 30, 3, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    weights = torch.randn(
        3, 2, device=torch_device, dtype=torch.float64, requires_grad=True
    )
    upstream = torch.randn(30, 3, device=torch_device, dtype=torch.float64)
    actual = gradient_module.accumulate(saved, weights)
    expected = reference(saved, weights)
    actual_grad = torch.autograd.grad(
        actual, (saved, weights), upstream, create_graph=True
    )
    expected_grad = torch.autograd.grad(
        expected, (saved, weights), upstream, create_graph=True
    )
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    (actual_second,) = torch.autograd.grad(actual_grad[0].sum(), weights)
    (expected_second,) = torch.autograd.grad(expected_grad[0].sum(), weights)
    torch.testing.assert_close(actual_second, expected_second, atol=0, rtol=0)


def test_weighted_gradient_cuda_memory_and_graph(gradient_module, torch_device):
    if torch_device.type != "cuda":
        pytest.skip("CUDA allocation and graph regression")
    saved = torch.randn(5, 64 * 5000, 3, device=torch_device)
    weights = torch.randn(5, 64, device=torch_device)

    def run():
        with torch.no_grad():
            return gradient_module.accumulate(saved, weights)

    def peak(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        result = fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before, result

    expected_peak, expected = peak(lambda: reference(saved, weights))
    actual_peak, actual = peak(run)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual_peak < expected_peak / 2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    weights.mul_(-2)
    graph.replay()
    torch.testing.assert_close(captured, reference(saved, weights), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [2, 3, 4, 5])
@pytest.mark.parametrize("output_values", [1, 2, 3, 31, 32, 33, 1025])
def test_weighted_gradient_short_outputs(
    gradient_module, torch_device, dtype, channels, output_values
):
    saved = (
        torch.tensor(
            [1e20, 1, -1e20, 2, -2][:channels], device=torch_device, dtype=dtype
        )[:, None, None]
        .expand(channels, output_values, 1)
        .contiguous()
    )
    weights = torch.ones(channels, 1, device=torch_device, dtype=dtype)
    with torch.no_grad():
        expected = reference(saved, weights)
        actual = gradient_module.accumulate(saved, weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
