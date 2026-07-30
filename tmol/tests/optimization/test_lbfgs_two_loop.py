import pytest
import torch

from tmol.optimization.lbfgs_armijo import lbfgs_two_loop
from tmol.tests.torch import requires_cuda


def _golden_two_loop(grad, dirs, stps):
    """The original implementation as the "gold standard" """
    m = dirs.shape[0]
    ro = 1.0 / (dirs * stps).sum(dim=1)  # rho_i = 1/(y_i . s_i)
    al = torch.zeros(m, dtype=grad.dtype, device=grad.device)
    result = -grad.clone()
    for i in range(m - 1, -1, -1):
        al[i] = ro[i] * torch.dot(stps[i], result)
        result = result - al[i] * dirs[i]
    for i in range(m):
        coeff = al[i] - ro[i] * torch.dot(dirs[i], result)
        result = result + coeff * stps[i]
    return result


def _random_inputs(N, m, dtype, device, seed=0):
    """grad/dirs/steps with guaranteed positive curvature
    (s_i dot y_i > 0)"""
    torch.manual_seed(seed)
    grad = torch.randn(N, dtype=dtype, device=device)
    S = torch.randn(m, N, dtype=dtype, device=device)  # stps
    Y = torch.randn(m, N, dtype=dtype, device=device)  # dirs
    dots = (S * Y).sum(dim=1)
    Y = torch.where((dots >= 0).unsqueeze(1), Y, -Y)
    Y = Y + 0.5 * S
    return grad.contiguous(), Y.contiguous(), S.contiguous()


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.parametrize("N,m", [(1, 1), (10, 1), (256, 5), (1024, 128), (5003, 64)])
def test_lbfgs_two_loop_matches_reference(N, m, dtype, torch_device):
    grad, dirs, stps = _random_inputs(N, m, dtype, torch_device)
    out = lbfgs_two_loop(grad, dirs, stps)
    ref = _golden_two_loop(grad, dirs, stps)
    atol, rtol = (1e-9, 1e-7) if dtype == torch.float64 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
@pytest.mark.benchmark(group="lbfgs_two_loop")
@requires_cuda
def test_lbfgs_two_loop_benchmark(benchmark, dtype):
    device = torch.device("cuda")
    N, m = 1024, 128
    grad, dirs, stps = _random_inputs(N, m, dtype, device)
    lbfgs_two_loop(grad, dirs, stps)  # warmup / load
    torch.cuda.synchronize()

    @benchmark
    def run():
        out = lbfgs_two_loop(grad, dirs, stps)
        torch.cuda.synchronize()
        return out
