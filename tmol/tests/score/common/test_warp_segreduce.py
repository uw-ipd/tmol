import pytest
import torch
import numpy

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename, cuda_if_available

from tmol.tests.torch import requires_cuda


@pytest.fixture
def warp_segreduce():
    return cpp_extension.load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["warp_segreduce.cpp", "warp_segreduce.cuda.cu"])
        ),
    )


@requires_cuda
def test_warp_segreduce_1(warp_segreduce):
    torch_device = torch.device("cuda")
    values = torch.ones((32,), dtype=torch.float32, device=torch_device)
    flags = torch.zeros((32,), dtype=torch.int32, device=torch_device)
    flags[1] = 1
    flags[2] = 1
    flags[4] = 1
    flags[8] = 1
    flags[16] = 1

    result = warp_segreduce.warp_segreduce_full(values, flags)

    gold_result = torch.zeros((32,), dtype=torch.float32)
    gold_result[1] = 1
    gold_result[2] = 2
    gold_result[4] = 4
    gold_result[8] = 8
    gold_result[16] = 16

    numpy.testing.assert_equal(gold_result.numpy(), result.cpu().numpy())


@requires_cuda
def test_warp_segreduce_vec3(warp_segreduce):
    torch_device = torch.device("cuda")
    values = torch.ones((32 * 3,), dtype=torch.float32, device=torch_device).view(32, 3)
    flags = torch.zeros((32,), dtype=torch.int32, device=torch_device)
    flags[1] = 1
    flags[2] = 1
    flags[4] = 1
    flags[8] = 1
    flags[16] = 1

    result = warp_segreduce.warp_segreduce_full_vec3(values, flags)

    gold_result = torch.zeros((32, 3), dtype=torch.float32)
    gold_result[1, :] = 1
    gold_result[2, :] = 2
    gold_result[4, :] = 4
    gold_result[8, :] = 8
    gold_result[16, :] = 16

    numpy.testing.assert_equal(gold_result.numpy(), result.cpu().numpy())


@pytest.mark.parametrize("n_repeats", [0, 1, 3, 10])
@pytest.mark.parametrize("n_vec3", [100, 500, 1000])
@pytest.mark.benchmark(group="warp_primatives_test")
@requires_cuda
def test_warp_segreduce_vec3_benchmark(benchmark, warp_segreduce, n_repeats, n_vec3):
    torch_device = torch.device("cuda")
    values = torch.ones((32 * 3,), dtype=torch.float32, device=torch_device).view(32, 3)
    flags = torch.zeros((32,), dtype=torch.int32, device=torch_device)
    flags[1] = 1
    flags[2] = 1
    flags[4] = 1
    flags[8] = 1
    flags[16] = 1
    values = values.repeat(n_vec3 * 1000, 1)
    flags = flags.repeat(n_vec3 * 1000)

    @benchmark
    def run():
        result = warp_segreduce.warp_segreduce_vec3_benchmark(values, flags, n_repeats)


@requires_cuda
def test_warp_segreduce_w_partial_warp(warp_segreduce):
    torch_device = torch.device("cuda")
    values = torch.ones((32,), dtype=torch.float32, device=torch_device)
    flags = torch.zeros((32,), dtype=torch.int32, device=torch_device)
    flags[1] = 1
    flags[2] = 1
    flags[4] = 1
    flags[8] = 1
    flags[16] = 1

    result = warp_segreduce.warp_segreduce_partial(values, flags)

    gold_result = torch.zeros((32,), dtype=torch.float32)
    gold_result[1] = 1
    gold_result[2] = 2
    gold_result[4] = 4
    gold_result[8] = 8
    gold_result[16] = 14

    numpy.testing.assert_equal(gold_result.numpy(), result.cpu().numpy())
