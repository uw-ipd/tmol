import math

import pytest
import torch
import torch.testing

from tmol.tests.torch import requires_cuda


@requires_cuda
@pytest.fixture
def geom():
    from tmol.tests.score.common.geom import _ext_cuda

    return _ext_cuda


@requires_cuda
def test_distance(geom):
    A = torch.zeros((10, 3), device="cuda")
    B = torch.ones((10, 3), device="cuda")
    V = torch.full((10,), math.sqrt(3), device="cuda")

    torch.testing.assert_close(geom.distance_V(A, B), V)
    torch.testing.assert_close(geom.distance_V_dV(A, B)[0], V)
