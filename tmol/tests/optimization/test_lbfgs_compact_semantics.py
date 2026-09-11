import pytest
import torch

from tmol.optimization import lbfgs_two_loop


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("negative_curvature", [False, True])
@pytest.mark.parametrize("differentiable", [False, True])
@pytest.mark.parametrize("layout", ["single", "batched", "broadcast"])
def test_compact_history_matches_dense_updates(
    dtype, negative_curvature, differentiable, layout, torch_device
):
    # Include coupled updates, a nonzero orthogonal pair, and an absent slot.
    steps = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        dtype=dtype,
        device=torch_device,
    )
    directions = torch.tensor(
        [[2, 1, 0, 0], [0, 4, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        dtype=dtype,
        device=torch_device,
    )
    if negative_curvature:
        directions[1].neg_()
    grad = torch.arange(1, 13, dtype=dtype, device=torch_device).reshape(3, 4)
    if layout == "single":
        grad = grad[0]
    grad.requires_grad_(differentiable)
    steps.requires_grad_(differentiable)
    directions.requires_grad_(differentiable)

    # Build H with dense rank-two updates, independently of the compact solves.
    identity = torch.eye(4, dtype=torch.float64, device=torch_device)
    hessian = identity
    for step, direction in zip(steps.double(), directions.double()):
        curvature = step.dot(direction)
        diagonal = torch.where(curvature == 0, 1, curvature)
        left = identity - torch.outer(step, direction) / diagonal
        hessian = (
            left @ hessian @ left.T
            + curvature * torch.outer(step, step) / diagonal.square()
        )
    expected = (-grad.double() @ hessian.T).to(dtype)
    if layout == "single":
        actual = lbfgs_two_loop(grad, directions, steps)
    else:
        batches = 3 if layout == "batched" else 1
        actual = lbfgs_two_loop(
            grad,
            directions[:, None].expand(-1, batches, -1),
            steps[:, None].expand(-1, batches, -1),
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if differentiable:
        inputs = grad, directions, steps
        actual_grads = torch.autograd.grad(actual.sum(), inputs)
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("batch,history_batch", [(1, 1), (2, 2), (2, 1)])
def test_compact_second_derivatives(torch_device, batch, history_batch):
    torch.manual_seed(491)
    grad = torch.randn(batch, 7, dtype=torch.float64, device=torch_device)
    steps = torch.randn(3, history_batch, 7, dtype=torch.float64, device=torch_device)
    directions = steps + 0.1 * torch.randn_like(steps)
    inputs = tuple(value.requires_grad_() for value in (grad, directions, steps))
    assert torch.autograd.gradgradcheck(lbfgs_two_loop, inputs, fast_mode=True)
