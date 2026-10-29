import pytest
import torch

from tmol.dock._batched_lbfgs import BatchedLBFGS


def _quadratic_evaluate(centers, scales):
    def evaluate(x):  # x [trial, pose, dof]
        delta = x - centers
        return (scales * delta**2).sum(-1), 2 * scales * delta

    return evaluate


@pytest.mark.parametrize("graph", [False, True])
def test_batched_lbfgs_minimizes_independent_poses(torch_device, graph):
    if graph and torch_device.type != "cuda":
        pytest.skip("CUDA graphs need CUDA")
    generator = torch.Generator().manual_seed(7)
    centers = torch.randn(5, 4, generator=generator).to(torch_device, torch.float64)
    scales = (1.0 + 20.0 * torch.rand(5, 4, generator=generator)).to(
        torch_device, torch.float64
    )
    x0 = torch.zeros_like(centers)
    minimizer = BatchedLBFGS(
        _quadratic_evaluate(centers, scales), x0, n_trials=4, atol=1e-12, graph=graph
    )
    first = minimizer.minimize(x0, 60)
    torch.testing.assert_close(first, centers, rtol=0.0, atol=1e-6)
    # The minimizer is reusable: a second run restarts from the new start.
    second = minimizer.minimize(x0 + 1.0, 60)
    torch.testing.assert_close(second, centers, rtol=0.0, atol=1e-6)
