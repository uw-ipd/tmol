import pytest
import torch

from tmol.optimization import lbfgs_two_loop
from tmol.tests import requires_cuda


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lbfgs_two_loop_batched_matches_reference(dtype, torch_device):
    cases = [_random_inputs(17, 4, dtype, torch_device, seed) for seed in range(3)]
    grad = torch.stack([case[0] for case in cases])
    dirs = torch.stack([case[1] for case in cases], dim=1)
    stps = torch.stack([case[2] for case in cases], dim=1)
    actual = lbfgs_two_loop(grad, dirs, stps)
    expected = torch.stack([_golden_two_loop(*case) for case in cases])
    tolerance = (1e-9, 1e-7) if dtype == torch.float64 else (1e-3, 1e-3)
    torch.testing.assert_close(actual, expected, atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "broadcast", "single"])
@pytest.mark.parametrize("differentiable", [False, True])
def test_lbfgs_two_loop_diagonal_history(dtype, layout, differentiable, torch_device):
    # Orthogonal updates give a known diagonal inverse Hessian. Two unused
    # history slots must leave both the direction and its derivative intact.
    batch = 1 if layout == "single" else 3
    history_batch = 1 if layout == "broadcast" else batch
    stride = 2 if layout in ("strided", "single") else 1
    steps = torch.zeros(5, history_batch, 7 * stride, dtype=dtype, device=torch_device)
    steps = steps[..., ::stride]
    diagonal = torch.tensor([2, 4, 8, 1, 1, 1, 1], dtype=dtype, device=torch_device)
    for index in range(3):
        steps[index, :, index] = 1
    directions = steps * diagonal
    grad = torch.arange(1, batch * 7 + 1, dtype=dtype, device=torch_device)
    grad = grad.reshape(batch, 7).requires_grad_(differentiable)
    result = lbfgs_two_loop(grad, directions, steps)
    torch.testing.assert_close(result, -grad / diagonal, atol=0, rtol=0)
    if differentiable:
        (derivative,) = torch.autograd.grad(result.sum(), grad)
        torch.testing.assert_close(
            derivative, (-1 / diagonal).expand_as(grad), atol=0, rtol=0
        )


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
