import pytest
import torch
import numpy

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename, cuda_if_available

from tmol.tests.torch import requires_cuda


@pytest.fixture
def warp_stride_reduce():
    return cpp_extension.load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["warp_stride_reduce.cpp", "warp_stride_reduce.cuda.cu"])
        ),
    )


@requires_cuda
def test_warp_stride_reduce_1(warp_stride_reduce):
    torch_device = torch.device("cuda")
    values = torch.arange(32, dtype=torch.float32, device=torch_device)

    result = warp_stride_reduce.warp_stride_reduce_1(values, 5)

    gold_result = torch.zeros((32,), dtype=torch.float32)
    gold_result[0] = 0 + 5 + 10 + 15 + 20 + 25 + 30
    gold_result[1] = 1 + 6 + 11 + 16 + 21 + 26 + 31
    gold_result[2] = 2 + 7 + 12 + 17 + 22 + 27
    gold_result[3] = 3 + 8 + 13 + 18 + 23 + 28
    gold_result[4] = 4 + 9 + 14 + 19 + 24 + 29

    numpy.testing.assert_equal(gold_result.numpy(), result.cpu().numpy())


@requires_cuda
def test_warp_stride_reduce_w_partial_warp(warp_stride_reduce):
    torch_device = torch.device("cuda")
    values = torch.arange(32, dtype=torch.float32, device=torch_device)

    result = warp_stride_reduce.warp_stride_reduce_2(values, 5)

    gold_result = torch.zeros((32,), dtype=torch.float32)
    gold_result[0] = 0 + 5 + 10 + 15 + 20 + 25
    gold_result[1] = 1 + 6 + 11 + 16 + 21 + 26
    gold_result[2] = 2 + 7 + 12 + 17 + 22 + 27
    gold_result[3] = 3 + 8 + 13 + 18 + 23 + 28
    gold_result[4] = 4 + 9 + 14 + 19 + 24 + 29

    numpy.testing.assert_equal(gold_result.numpy(), result.cpu().numpy())
