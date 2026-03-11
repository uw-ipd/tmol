import math

import pytest
import torch
import torch.testing

from tmol.tests.torch import requires_cuda


@pytest.fixture
def geom():
    from tmol._load_ext import load_module

    # The module_name "_ext_cuda" must match PYBIND11_MODULE(_ext_cuda, m) in
    # geom.cu; modulename() is a no-op here (no dots to replace).
    return load_module(
        "_ext_cuda",
        __file__,
        "geom.cu",
        "tmol.tests.score.common.geom._ext_cuda",
    )


@requires_cuda
def test_distance(geom):
    A = torch.zeros((10, 3), device="cuda")
    B = torch.ones((10, 3), device="cuda")
    V = torch.full((10,), math.sqrt(3), device="cuda")

    torch.testing.assert_close(geom.distance_V(A, B), V)
    torch.testing.assert_close(geom.distance_V_dV(A, B)[0], V)
