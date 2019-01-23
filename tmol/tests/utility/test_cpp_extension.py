import pytest

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename

from tmol.tests.torch import requires_cuda


def test_pure_cpp_extension():
    """Pure cpp extension builds cpu-only interface.

    Pure cpp extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU tensors, and raises error on CUDA tensors.
    """
    extension = load(
        modulename(f"{__name__}.pure_cpp_extension"),
        relpaths(__file__, "pure_cpp_extension.cpp"),
    )

    assert extension.sum(torch.ones(10)) == 10

    if torch.cuda.is_available():
        with pytest.raises(TypeError):
            extension.sum(torch.ones(10, device="cuda"))


@requires_cuda
def test_cuda_extension():
    """Hybrid cpp/cuda extension defined in .cu file builds combined interface.

    Hybrid extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU and CUDA tensors.
    """
    extension = load(
        modulename(f"{__name__}.cuda_extension"),
        relpaths(__file__, "cuda_extension.cu"),
    )

    assert extension.sum(torch.ones(10, device="cpu")) == 10
    assert extension.sum(torch.ones(10, device="cuda")) == 10
