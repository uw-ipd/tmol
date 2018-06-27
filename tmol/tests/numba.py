import pytest

import importlib

import numba
import numba.cuda


@pytest.fixture
def numba_cudasim():
    old_numba_cudasim = numba.config.ENABLE_CUDASIM

    numba.config.ENABLE_CUDASIM = 1
    importlib.reload(numba.cuda)

    yield numba.cuda

    numba.config.ENABLE_CUDASIM = old_numba_cudasim
    importlib.reload(numba.cuda)
