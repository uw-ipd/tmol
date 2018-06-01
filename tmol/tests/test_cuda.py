import pytest
import torch
from .torch import requires_cuda


@requires_cuda
def test_torch_cuda_is_available():
    import torch
    assert torch.cuda.is_available()


@requires_cuda
def test_torch_cuda_smoke():
    import torch

    rs = (100, 100)
    a = torch.rand(rs)
    b = torch.rand(rs)

    c = a.cuda() @ b.cuda()

    torch.testing.assert_allclose(a @ b, c.cpu())


@pytest.mark.parametrize(
    "dtype",
    [torch.float, torch.double],
    ids=("single", "double"),
)
@pytest.mark.benchmark(
    group="float_perf",
)
def test_float_perf(benchmark, torch_device, dtype):
    import torch

    test_size = 1500
    test_coords = torch.rand((test_size, 3), device=torch_device, dtype=dtype)

    @benchmark
    def sum_distances():
        delta = (
            test_coords.reshape((test_size, 1, 3)) -
            test_coords.reshape((1, test_size, 3))
        )

        dists = delta.norm(dim=-1)

        return float(dists.sum())
