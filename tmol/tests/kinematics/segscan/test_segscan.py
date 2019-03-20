import pytest

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename

from tmol.tests.torch import requires_cuda


@requires_cuda
def test_segscan(benchmark):
    """CUDA segscan code
    """
    extension = load(modulename(f"{__name__}.cuda"), relpaths(__file__, "segscan.cu"))

    x = torch.ones(1000000, dtype=torch.float32)
    segs = torch.tensor([4, 7, 35, 273, 1129, 43567, 143678, 567778], dtype=torch.int)

    xcuda = x.to(device="cuda")
    segscuda = segs.to(device="cuda")

    y = extension.segscan(x, segs)
    ycuda = extension.segscan(xcuda, segscuda)

    torch.testing.assert_allclose(ycuda.to(device="cpu"), y)


@requires_cuda
@pytest.mark.benchmark
def test_segscan_cudabench(benchmark):
    """CUDA segscan benchmark
    """
    extension = load(modulename(f"{__name__}.cuda"), relpaths(__file__, "segscan.cu"))

    x = torch.ones(10000000, dtype=torch.float32)
    segs = torch.tensor([4, 7, 35, 273, 1129, 43567, 143678, 567778], dtype=torch.int)
    xcuda = x.to(device="cuda")
    segscuda = segs.to(device="cuda")

    def cuda_segscan():
        return extension.segscan(xcuda, segscuda)

    benchmark(cuda_segscan)


@requires_cuda
@pytest.mark.benchmark
def test_segscan_cpubench(benchmark):
    """CPU segscan baseline benchmark
    """
    extension = load(modulename(f"{__name__}.cuda"), relpaths(__file__, "segscan.cu"))

    x = torch.ones(10000000, dtype=torch.float32)
    segs = torch.tensor([4, 7, 35, 273, 1129, 43567, 143678, 567778], dtype=torch.int)

    def cpu_segscan():
        return extension.segscan(x, segs)

    benchmark(cpu_segscan)
