import numpy
import math

import pytest

from tmol.tests.torch import requires_cuda

import torch.cuda
import numba.cuda

from tmol.types.torch import _torch_dtype_mapping


@requires_cuda
def test_array_adaptor():
    # Import in test due to expected assertion failure on numba upgrade
    from tmol.utility.numba import as_cuda_array, is_cuda_array

    for dt in set(_torch_dtype_mapping.values()):
        if dt == torch.int8:
            # Skip tests of int8, not official supported by pytorch
            continue

        cput = torch.arange(10).to(dt)
        npt = cput.numpy()

        assert not is_cuda_array(cput)

        with pytest.raises(ValueError):
            as_cuda_array(cput)

        cudat = cput.to(device="cuda")
        assert is_cuda_array(cudat)

        numba_view = as_cuda_array(cudat)
        assert isinstance(numba_view, numba.cuda.devicearray.DeviceNDArray)
        assert numba_view.dtype == npt.dtype
        assert numba_view.strides == npt.strides
        assert numba_view.shape == cudat.shape
        # Pass back to cuda from host for fp16 comparisons
        assert (cudat == torch.tensor(numba_view.copy_to_host()
                                      ).to("cuda")).all()

        cudat[:5] = math.pi
        # Pass back to cuda from host for fp16 comparisons
        assert (cudat == torch.tensor(numba_view.copy_to_host()
                                      ).to("cuda")).all()

        strided_cudat = cudat[::2]
        strided_numba_view = as_cuda_array(strided_cudat)
        with pytest.raises((TypeError, ValueError)):
            # Bug with copies of strided data device->host
            assert (
                strided_cudat.to("cpu") == torch.tensor(
                    strided_numba_view.copy_to_host()
                )
            ).all()

        result_buffer = numpy.empty(10, dtype=strided_numba_view.dtype)
        result_view = result_buffer[::2]
        strided_numba_view.copy_to_host(result_view)
        # Pass back to cuda from host for fp16 comparisons
        assert (strided_cudat == torch.tensor(result_view).to("cuda")).all()


@requires_cuda
@pytest.mark.skipif(
    len(numba.cuda.devices.gpus) < 2,
    reason="Requires multiple cuda devices.",
)
def test_active_device():
    # Import in test due to expected assertion failure on numba upgrade
    from tmol.utility.numba import as_cuda_array, is_cuda_array

    # Should fail if the tensor device id is not the current numba context.
    cudat = torch.arange(10).to(device=torch.device("cuda", 1))
    assert is_cuda_array(cudat)
    with pytest.raises(ValueError):
        as_cuda_array(cudat)
