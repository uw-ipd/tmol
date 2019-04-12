import attr
import pytest
import torch
import math

from tmol.types.torch import Tensor, TensorCollection
import tmol.utility.tensor.compiled as tutc
import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Dummy:
    foo: TensorCollection(torch.float)[2]


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
    tcoll_cpp = tutc.create_tensor_collection(tcoll)
    tsum = tensor_collection.sum_tensor_collection(tcoll_cpp)
    expected = tcoll[0] + tcoll[1]
    torch.testing.assert_allclose(expected, tsum)


def test_tensor_collection_validation(torch_device):
    tensor = torch.zeros([2, 3], dtype=torch.float, device=torch_device)
    tensor_list = [tensor]
    tcollection = tutc.create_tensor_collection(tensor_list)
    assert tcollection.device == torch_device
    d = Dummy(foo=tcollection)

    assert len(tcollection) == 1

    assert tcollection.shape(0) == (2, 3)
