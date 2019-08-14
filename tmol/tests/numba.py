import pytest

import importlib

import numba
import numba.cuda

import contextlib

def is_jit_available() :
    import os
    return os.environ["NUMBA_DISABLE_JIT"] == 0

jit_available = is_jit_available()
requires_numba_jit = pytest.mark.skipif(not jit_available, reason="Requires JIT")

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
