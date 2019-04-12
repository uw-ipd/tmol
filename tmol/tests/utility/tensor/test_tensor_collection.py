import attr
import pytest
import torch

from tmol.types.torch import TensorCollection
from tmol.types.attrs import ValidateAttrs

import tmol.utility.tensor.compiled as tutc
import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Dummy(ValidateAttrs):
    foo: TensorCollection(torch.float)[:, :]


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


def test_tensor_collection_validation1(torch_device):
    tensor = torch.zeros([2, 3], dtype=torch.float, device=torch_device)
    tensor_list = [tensor]
    tcollection = tutc.create_tensor_collection(tensor_list)
    assert tcollection.device == torch_device

    # Test that TensorCollection validates properly
    Dummy(foo=tcollection)

    assert len(tcollection) == 1

    assert tcollection.shape(0) == (2, 3)


def test_tensor_collection_validation2(torch_device):

    tctypes = [
        TensorCollection(torch.float)[:],
        TensorCollection(torch.float)[:, :],
        TensorCollection(torch.float)[:, :, :],
        TensorCollection(torch.float)[:, :, :, :],
    ]

    for ndim in range(1, 5):
        tensor = torch.zeros([3] * ndim, dtype=torch.float, device=torch_device)
        tensor_list = [tensor]
        tcollection = tutc.create_tensor_collection(tensor_list)

        assert tcollection.device == torch_device
        assert tctypes[ndim - 1].validate(tcollection)


def test_tensor_collection_validation_failure(torch_device):

    tctypes = [
        TensorCollection(torch.float)[4],
        TensorCollection(torch.float)[4, 4],
        TensorCollection(torch.float)[4, 4, 4],
        TensorCollection(torch.float)[4, 4, 4, 4],
    ]

    emsg = [
        "expected TCollection element 0 of shape Shape(dims=(Dim(size=4),)), but its shape is torch.Size([3])",
        "expected TCollection element 0 of shape Shape(dims=(Dim(size=4), Dim(size=4))), but its shape is torch.Size([3, 3])",
        "expected TCollection element 0 of shape Shape(dims=(Dim(size=4), Dim(size=4), Dim(size=4))), but its shape is torch.Size([3, 3, 3])",
        "expected TCollection element 0 of shape Shape(dims=(Dim(size=4), Dim(size=4), Dim(size=4), Dim(size=4))), but its shape is torch.Size([3, 3, 3, 3])",
    ]

    for ndim in range(1, 5):
        tensor = torch.zeros([3] * ndim, dtype=torch.float, device=torch_device)
        tensor_list = [tensor]
        tcollection = tutc.create_tensor_collection(tensor_list)

        assert tcollection.device == torch_device
        try:
            tctypes[ndim - 1].validate(tcollection)
            assert False
        except ValueError as e:
            assert str(e) == emsg[ndim - 1]
