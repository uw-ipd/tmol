import torch
import pytest

from tmol.pose.compiled.apsp_ops import stacked_apsp
from tmol.tests.torch import requires_cuda, zero_padded_counts


def test_all_pairs_shortest_paths_simple_path_graph1(torch_device):
    weights = torch.full((1, 32, 32), -1, dtype=torch.int32, device=torch_device)
    arange32 = torch.arange(32, dtype=torch.int64, device=torch_device)
    weights[0, arange32, arange32] = 0
    weights[0, arange32[1:], arange32[:-1]] = 1
    weights[0, arange32[:-1], arange32[1:]] = 1

    stacked_apsp(weights, -1)

    weights_gold = torch.full((1, 32, 32), -1, dtype=torch.int32, device=torch_device)
    for i in range(32):
        weights_gold[0, i] = torch.abs(arange32 - i)

    torch.testing.assert_close(weights, weights_gold)


def test_all_pairs_shortest_paths_simple_path_graph1_w_cutoff():
    torch_device = torch.device("cpu")
    weights = torch.full((1, 32, 32), 6, dtype=torch.int32, device=torch_device)
    arange32 = torch.arange(32, dtype=torch.int64, device=torch_device)
    weights[0, arange32, arange32] = 0
    weights[0, arange32[1:], arange32[:-1]] = 1
    weights[0, arange32[:-1], arange32[1:]] = 1

    stacked_apsp(weights, 6)

    weights_gold = torch.full((1, 32, 32), 6, dtype=torch.int32, device=torch_device)
    for i in range(32):
        weights_gold[0, i] = torch.min(
            torch.abs(arange32 - i), torch.full((32,), 6, dtype=torch.int32)
        )

    torch.testing.assert_close(weights, weights_gold)


def test_all_pairs_shortest_paths_simple_path_graph2(torch_device):
    weights = torch.full((1, 64, 64), -1, dtype=torch.int32, device=torch_device)
    arange64 = torch.arange(64, dtype=torch.int64, device=torch_device)
    weights[0, arange64, arange64] = 0
    weights[0, arange64[1:], arange64[:-1]] = 1
    weights[0, arange64[:-1], arange64[1:]] = 1

    stacked_apsp(weights, -1)

    weights_gold = torch.full((1, 64, 64), -1, dtype=torch.int32, device=torch_device)
    for i in range(64):
        weights_gold[0, i] = torch.abs(arange64 - i)

    torch.testing.assert_close(weights, weights_gold)


def test_all_pairs_shortest_paths_big_simple_path_graph(torch_device):
    n_nodes = 1000
    n_graphs = 5

    weights = torch.full(
        (n_graphs, n_nodes, n_nodes), -1, dtype=torch.int32, device=torch_device
    )
    arange_n_nodes = torch.arange(n_nodes, dtype=torch.int64, device=torch_device)
    weights[:, arange_n_nodes, arange_n_nodes] = 0
    weights[:, arange_n_nodes[1:], arange_n_nodes[:-1]] = 1
    weights[:, arange_n_nodes[:-1], arange_n_nodes[1:]] = 1

    stacked_apsp(weights, -1)

    weights_gold = torch.full(
        (n_graphs, n_nodes, n_nodes), -1, dtype=torch.int32, device=torch_device
    )
    for i in range(n_nodes):
        weights_gold[0, i] = torch.abs(arange_n_nodes - i)
    weights_gold[1:] = weights_gold[0:1]

    torch.testing.assert_close(weights, weights_gold)


@requires_cuda
def test_all_pairs_shortest_paths_w_off_diagonal_bonds():
    torch_device_cpu = torch.device("cpu")
    torch_device_cuda = torch.device("cuda")

    n_nodes = 128
    n_graphs = 5

    weights = torch.full(
        (n_graphs, n_nodes, n_nodes), -1, dtype=torch.int32, device=torch_device_cpu
    )
    arange_n_nodes = torch.arange(n_nodes, dtype=torch.int64, device=torch_device_cpu)
    weights[:, arange_n_nodes, arange_n_nodes] = 0
    weights[:, arange_n_nodes[1:], arange_n_nodes[:-1]] = 1
    weights[:, arange_n_nodes[:-1], arange_n_nodes[1:]] = 1
    weights[:, 10, 20] = 1
    weights[:, 20, 10] = 1

    weights_cuda = weights.clone().to(torch_device_cuda)

    stacked_apsp(weights, -1)
    stacked_apsp(weights_cuda, -1)

    torch.testing.assert_close(weights, weights_cuda.cpu())

    cuda_result_cpu = weights_cuda.cpu().numpy()
    for i in range(n_nodes):
        for j in range(n_nodes):
            assert cuda_result_cpu[0, i, j] == cuda_result_cpu[0, j, i]


@requires_cuda
def test_all_pairs_shortest_paths_w_off_diagonal_bonds_and_threshold():
    # note that there is an expected difference in behavior here:
    # the cuda implementation does not respect the threshold and
    # so reports the actual path distance; it has to be "corrected"
    # to match the CPU's computed result
    torch_device_cpu = torch.device("cpu")
    torch_device_cuda = torch.device("cuda")

    n_nodes = 128
    n_graphs = 5

    weights = torch.full(
        (n_graphs, n_nodes, n_nodes), -1, dtype=torch.int32, device=torch_device_cpu
    )
    arange_n_nodes = torch.arange(n_nodes, dtype=torch.int64, device=torch_device_cpu)
    weights[:, arange_n_nodes, arange_n_nodes] = 0
    weights[:, arange_n_nodes[1:], arange_n_nodes[:-1]] = 1
    weights[:, arange_n_nodes[:-1], arange_n_nodes[1:]] = 1
    weights[:, 10, 20] = 1
    weights[:, 20, 10] = 1

    weights_cuda = weights.clone().to(torch_device_cuda)

    stacked_apsp(weights, 6)
    stacked_apsp(weights_cuda, 6)
    weights_cuda[weights_cuda > 6] = 6

    torch.testing.assert_close(weights, weights_cuda.cpu())

    cuda_result_cpu = weights_cuda.cpu().numpy()
    for i in range(n_nodes):
        for j in range(n_nodes):
            assert cuda_result_cpu[0, i, j] == cuda_result_cpu[0, j, i]


@pytest.mark.parametrize("n_nodes", zero_padded_counts([30, 100, 300]))
@pytest.mark.parametrize("n_graphs", zero_padded_counts([1, 3, 10, 30]))
@pytest.mark.benchmark(group="all_pairs_shortest_paths")
def test_all_pairs_shortest_paths_benchmark(benchmark, torch_device, n_graphs, n_nodes):
    n_nodes = int(n_nodes)
    n_graphs = int(n_graphs)

    weights = torch.full(
        (n_graphs, n_nodes, n_nodes), -1, dtype=torch.int32, device=torch_device
    )
    arange_n_nodes = torch.arange(n_nodes, dtype=torch.int64, device=torch_device)
    weights[:, arange_n_nodes, arange_n_nodes] = 0
    weights[:, arange_n_nodes[1:], arange_n_nodes[:-1]] = 1
    weights[:, arange_n_nodes[:-1], arange_n_nodes[1:]] = 1

    weights_gold = weights.clone()

    @benchmark
    def run_apsp():
        weights[:] = weights_gold
        stacked_apsp(weights, 6)
