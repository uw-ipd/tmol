import math

import pytest
import torch
import torch.testing

from tmol.utility.cpp_extension import load, relpaths, modulename
from tmol.tests.torch import requires_cuda


@requires_cuda
@pytest.fixture
def geom():
    return load(modulename(__name__) + ".geom", relpaths(__file__, "geom.cu"))


@requires_cuda
def test_distance(geom):

    A = torch.zeros((10, 3), device="cuda")
    B = torch.ones((10, 3), device="cuda")
    V = torch.full((10,), math.sqrt(3), device="cuda")

    torch.testing.assert_allclose(geom.distance_V(A, B), V)
    torch.testing.assert_allclose(geom.distance_V_dV(A, B)[0], V)
