import pytest
import torch

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
        verbose=True,
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

    result = warp_segreduce.warp_segreduce_1(values, flags)
    print(result)
