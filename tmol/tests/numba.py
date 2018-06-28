import pytest

import importlib

import numba
import numba.cuda

import contextlib


@contextlib.contextmanager
def with_cudasim():
    """Rebinds `numba.cuda` to the simulator for the context."""
    old_numba_dict = dict(numba.cuda.__dict__)
    old_numba_cudasim = numba.config.ENABLE_CUDASIM

    numba.config.ENABLE_CUDASIM = 1
    importlib.reload(numba.cuda)

    try:
        yield numba.cuda
    finally:
        numba.config.ENABLE_CUDASIM = old_numba_cudasim
        importlib.reload(numba.cuda)

        numba.cuda.__dict__.update(old_numba_dict)


@pytest.fixture(scope="function")
def numba_cudasim():
    with with_cudasim():
        yield numba.cuda


@pytest.fixture(scope="function")
def numba_cuda_or_cudasim():
    if numba.cuda.is_available():
        yield numba.cuda
    else:
        with with_cudasim():
            yield numba.cuda
