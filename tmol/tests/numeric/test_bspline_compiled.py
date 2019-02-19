import torch
import numpy
import pytest
from tmol.numeric.bspline_compiled import compiled


def test_compiled_bspline(torch_device):
    x = torch.zeros((3, 4, 5), dtype=torch.float)
    y = compiled.computeCoeffs(x)
    print(y)
    assert False
