import numpy
import math

import pytest

from tmol.tests.torch import requires_cuda

import torch

import torch.cuda
import numba.cuda

from numba.cuda import as_cuda_array, is_cuda_array

import tmol.utility.numba  # noqa


@requires_cuda
def test_array_adaptor():
    """Torch __cuda_array_adaptor__ monkeypatch, loaded via tmol.utility.numba."""

    torch_dtypes = [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]

    for dt in torch_dtypes:

        # CPU tensors of all types do not register as cuda arrays,
        # attempts to convert raise a type error.
        cput = torch.arange(10).to(dt)
        npt = cput.numpy()

        assert not is_cuda_array(cput)
        with pytest.raises(TypeError):
            as_cuda_array(cput)

        # Any cuda tensor is a cuda array.
        cudat = cput.to(device="cuda")
        assert is_cuda_array(cudat)

        numba_view = as_cuda_array(cudat)
        assert isinstance(numba_view, numba.cuda.devicearray.DeviceNDArray)

        # The reported type of the cuda array matches the numpy type of the cpu tensor.
        assert numba_view.dtype == npt.dtype
        assert numba_view.strides == npt.strides
        assert numba_view.shape == cudat.shape

        # Pass back to cuda from host for all equality checks below, needed for
        # float16 comparisons, which aren't supported cpu-side.

        # The data is identical in the view.
        assert (cudat == torch.tensor(numba_view.copy_to_host()).to("cuda")).all()

        # Writes to the torch.Tensor are reflected in the numba array.
        cudat[:5] = math.pi
        assert (cudat == torch.tensor(numba_view.copy_to_host()).to("cuda")).all()

        # Strided tensors are supported.
        strided_cudat = cudat[::2]
        strided_numba_view = as_cuda_array(strided_cudat)

        # Previous bug with numba (~0.40) copies of strided data device->host
        assert (
            strided_cudat.to("cpu") == torch.tensor(strided_numba_view.copy_to_host())
        ).all()

        # Can workaround with a strided result buffer for the copy.
        result_buffer = numpy.empty(10, dtype=strided_numba_view.dtype)
        result_view = result_buffer[::2]
        strided_numba_view.copy_to_host(result_view)
        assert (strided_cudat == torch.tensor(result_view).to("cuda")).all()


@requires_cuda
@pytest.mark.skipif(
    not numba.cuda.is_available() or len(numba.cuda.devices.gpus) < 2,
    reason="Requires multiple cuda devices.",
)
def test_active_device():
    """'as_cuda_array' tensor device must match active numba context."""

    # Both torch/numba default to device 0 and can interop freely
    cudat = torch.arange(10, device="cuda")
    assert cudat.device.index == 0
    assert as_cuda_array(cudat)

    # Tensors on non-default device raise api error if converted
    cudat = torch.arange(10, device=torch.device("cuda", 1))
    with pytest.raises(numba.cuda.driver.CudaAPIError):
        as_cuda_array(cudat)

    # but can be converted when switching to the device's context
    with numba.cuda.devices.gpus[cudat.device.index]:
        assert as_cuda_array(cudat)
