import pytest
import torch

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename


@pytest.fixture
def tensor_collection():
    return cpp_extension.load(
        modulename(__name__), relpaths(__file__, "tensor_collection.cpp"), verbose=True
    )


def test_tensor_collection(tensor_collection):
    tcoll = [
        torch.arange(4, dtype=torch.float).reshape(2, 2),
        torch.arange(1, 5, dtype=torch.float).reshape(2, 2),
    ]
    tsum = tensor_collection.sum_tensor_collection(tcoll)
    expected = tcoll[0] + tcoll[1]
    torch.testing.assert_allclose(expected, tsum)
