import attr
import pytest
import torch
import math

from tmol.types.torch import Tensor, TensorCollection
import tmol.utility.tensor.compiled as tutc


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Dummy:
    foo: TensorCollection(torch.float)[2]


def test_tensor_collection_validation(torch_device):
    tensor = torch.zeros([2, 3], dtype=torch.float, device=torch_device)
    tensor_list = [tensor]
    tcollection = tutc.create_tensor_collection2(tensor_list)
    assert tcollection.device == torch_device
    d = Dummy(foo=tcollection)

    assert len(tcollection) == 1

    assert tcollection.shape(0) == (2, 3)
