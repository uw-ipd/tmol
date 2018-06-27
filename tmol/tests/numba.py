import pytest

import importlib

import numba
import numba.cuda


@pytest.fixture(scope="function")
def numba_cudasim():
    old_numba_cudasim = numba.config.ENABLE_CUDASIM

    numba.config.ENABLE_CUDASIM = 1
    importlib.reload(numba.cuda)

    yield numba.cuda

    numba.config.ENABLE_CUDASIM = old_numba_cudasim
    importlib.reload(numba.cuda)


@pytest.fixture(scope="function")
def numba_cuda_or_cudasim():
    if numba.cuda.is_available():
        yield numba.cuda
    else:
        old_numba_cudasim = numba.config.ENABLE_CUDASIM

        numba.config.ENABLE_CUDASIM = 1
        importlib.reload(numba.cuda)

        yield numba.cuda

        numba.config.ENABLE_CUDASIM = old_numba_cudasim
        importlib.reload(numba.cuda)
