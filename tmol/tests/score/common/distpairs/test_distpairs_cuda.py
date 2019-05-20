from tmol.utility.cpp_extension import load, relpaths, modulename
from tmol.tests.torch import requires_cuda

import torch
import numpy
import pytest


@requires_cuda
@pytest.fixture
def dist_check(scope="session"):
    import tmol.tests.score.common.distpairs.distpairs as distpairs

    # return load(modulename(__name__)+".distpairs", relpaths(__file__, "dist.cu"))
    return distpairs


@requires_cuda
def test_triu_distance_check_gpu(dist_check):
    gpu = torch.device("cuda:0")
    coords = torch.zeros((64, 3), dtype=torch.float32)
    for i in range(64):
        for j in range(3):
            coords[i, j] = i if j == 0 else 0
    # print("coords")
    # print(coords)
    coords = coords.to(gpu)
    nearby, nearby_scan, i_ind, j_ind = dist_check.triu_distpairs(coords, 6.5)
    # print("nearby")
    # print(nearby)
    # numpy.set_printoptions(threshold=numpy.inf)
    # print("nearby_scan")
    # print(nearby_scan.cpu().numpy())
    #
    # print("i_ind")
    # print(i_ind.cpu().numpy())
    #
    # print("nearby_j")
    # print(j_ind.cpu().numpy())

    gold_nearby = torch.zeros((64 * 63 // 2,), dtype=torch.int)
    gold_i_ind = torch.zeros((64 * 63 // 2,), dtype=torch.int)
    gold_j_ind = torch.zeros((64 * 63 // 2,), dtype=torch.int)
    count = 0
    count_in_range = 0
    for i in range(64):
        for j in range(i + 1, 64):
            gold_nearby[count] = int(j - i < 6.5)
            if j - i < 6.5:
                gold_i_ind[count_in_range] = i
                gold_j_ind[count_in_range] = j
                count_in_range += 1
            count = count + 1
    nearby = nearby.cpu()
    i_ind = i_ind.cpu()
    j_ind = j_ind.cpu()
    torch.testing.assert_allclose(nearby, gold_nearby)
    torch.testing.assert_allclose(i_ind[:count_in_range], gold_i_ind[:count_in_range])
    torch.testing.assert_allclose(j_ind[:count_in_range], gold_j_ind[:count_in_range])
    # for count in range(nearby.shape[0]):
    #     if nearby[count] != gold[count]:
    #         print("x count", count, "nearby[count]", nearby[count], "gold", gold[count])
    #     else :
    #         print("count", count, "nearby[count]", nearby[count], "gold", gold[count])
