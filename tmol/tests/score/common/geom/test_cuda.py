import math

import pytest
import torch
import torch.testing

from tmol.tests.torch import requires_cuda
from tmol._load_ext import ensure_compiled_or_jit


@pytest.fixture
def geom():
    if ensure_compiled_or_jit():
        from tmol.utility.cpp_extension import load, relpaths

        # The name passed to load() must match PYBIND11_MODULE(name, m) in
        # geom.cu, which is "_ext_cuda".  Using modulename(__name__) here
        # would produce "tmol_tests_score_common_geom_test_cuda", causing
        # "PyInit_<that>" to be looked up in the .so instead of "PyInit__ext_cuda".
        _ext_cuda = load("_ext_cuda", relpaths(__file__, "geom.cu"))
    else:
        from tmol.tests.score.common.geom import _ext_cuda

    return _ext_cuda


@requires_cuda
def test_distance(geom):
    A = torch.zeros((10, 3), device="cuda")
    B = torch.ones((10, 3), device="cuda")
    V = torch.full((10,), math.sqrt(3), device="cuda")

    torch.testing.assert_close(geom.distance_V(A, B), V)
    torch.testing.assert_close(geom.distance_V_dV(A, B)[0], V)
