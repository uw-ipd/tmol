import pytest

import torch

from tmol.utility.cpp_extension import (
    load,
    relpaths,
    modulename,
    cuda_if_available,
    get_prebuild_extensions,
)

from tmol.tests.torch import requires_cuda


def test_pure():
    """Pure cpp extension builds cpu-only interface.

    Pure cpp extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU tensors, and raises error on CUDA tensors.
    """
    extension = load(modulename(f"{__name__}.pure"), relpaths(__file__, "pure.cpp"))

    assert extension.sum(torch.ones(10)) == 10

    if torch.cuda.is_available():
        with pytest.raises(TypeError):
            extension.sum(torch.ones(10, device="cuda"))


@requires_cuda
def test_cuda():
    """Hybrid cpp/cuda extension defined in .cu file builds combined interface.

    Hybrid extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU and CUDA tensors.
    """
    extension = load(modulename(f"{__name__}.cuda"), relpaths(__file__, "cuda.cu"))

    assert extension.sum(torch.ones(10, device="cpu")) == 10
    assert extension.sum(torch.ones(10, device="cuda")) == 10


def test_hybrid(torch_device):
    """Hybrid cpp/cuda defined in .cpp/.cu files builds combined interface.

    Hybrid extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU and CUDA tensors.
    """
    extension = load(
        modulename(f"{__name__}.hybrid"),
        cuda_if_available(
            relpaths(__file__, ["hybrid.pybind.cpp", "hybrid.cpp", "hybrid.cu"])
        ),
    )

    assert extension.sum(torch.ones(10, device=torch_device)) == 10


def test_hybrid_nocuda():
    """Hybrid cpp/cuda extension w/o cuda support compiled cpu-only interface.

    Hybrid extension compiled with cpu-only sources or without cuda uses
    "WITH_CUDA" define to compile cpu-only pybind interface. Attempt to call
    through cuda interface then raises TypeError.
    """

    extension_nocuda = load(
        modulename(f"{__name__}.hybrid.nocuda"),
        relpaths(__file__, ["hybrid.pybind.cpp", "hybrid.cpp"]),
    )

    assert extension_nocuda.sum(torch.ones(10, device="cpu")) == 10
    if torch.cuda.is_available():
        with pytest.raises(TypeError):
            extension_nocuda.sum(torch.ones(10, device="cuda"))


def test_get_prebuild_extensions_smoke():
    """
    Just make sure this function runs. Should probably inspect the output for validity with a better test
    """
    get_prebuild_extensions()
