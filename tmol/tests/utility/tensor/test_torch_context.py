from concurrent.futures import ThreadPoolExecutor

import pytest
import torch


@pytest.fixture(scope="module")
def context_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA context allocator")
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        ["torch_context.pybind.cpp", "torch_context.cuda.cu"],
        "tmol.tests.utility.tensor._torch_context",
    )


def test_cuda_context_accounts_for_temporary_storage(context_module):
    values = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
    context_module.copy(values)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = context_module.copy(values)
    peak = torch.cuda.max_memory_allocated() - before
    torch.testing.assert_close(result, values, atol=0, rtol=0)
    assert peak >= 2 * values.numel() * values.element_size()
    assert context_module.copy(values[:0]).numel() == 0


def test_cuda_context_capture_preserves_stream_order(context_module):
    values = torch.arange(1024, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context_module.copy(values)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = context_module.copy(values)
    for value in (7, -13, 29):
        values.fill_(value)
        graph.replay()
        torch.testing.assert_close(result, values, atol=0, rtol=0)


def test_cuda_context_concurrent_streams(context_module):
    streams = [torch.cuda.Stream() for _ in range(8)]

    def run(index):
        stream = streams[index]
        with torch.cuda.stream(stream):
            values = torch.full((4096,), index, dtype=torch.int32, device="cuda")
            for _ in range(8):
                result = context_module.copy(values)
        stream.synchronize()
        torch.testing.assert_close(result, values, atol=0, rtol=0)

    with ThreadPoolExecutor(max_workers=len(streams)) as pool:
        list(pool.map(run, range(len(streams))))


def test_cuda_context_preserves_caller_device(context_module):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    original_device = torch.cuda.current_device()
    for target_device in (0, 1):
        values = torch.arange(1024, device=f"cuda:{target_device}", dtype=torch.int32)
        result = context_module.copy(values)
        assert result.device == values.device
        assert torch.cuda.current_device() == original_device
        torch.testing.assert_close(result, values, atol=0, rtol=0)
