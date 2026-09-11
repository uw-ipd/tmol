import copy

import pytest
import torch

from tmol.optimization import LBFGS_Armijo


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lengths", [(7, 7), (5, 9)])
def test_checkpoint_resume_preserves_trajectory(torch_device, dtype, lengths):
    device = torch.device(torch_device)
    initial = torch.linspace(-2, 2, sum(lengths), device=device, dtype=dtype)
    target = initial.sin()
    curvature = torch.logspace(-2, 2, sum(lengths), device=device, dtype=dtype)
    segments = torch.repeat_interleave(
        torch.arange(2, device=device), torch.tensor(lengths, device=device)
    )

    def optimizer_for(x):
        return LBFGS_Armijo(
            [x],
            segment_ids=segments,
            history_size=3,
            max_iter=6,
            fixed_iterations=True,
            rtol=0,
            atol=0,
            gradtol=0,
        )

    def step(x, optimizer):
        losses = []

        def closure():
            optimizer.zero_grad()
            terms = curvature * (x - target).square()
            energies = torch.stack([part.sum() for part in terms.split(lengths)])
            energies.sum().backward()
            losses.append(energies.detach())
            return energies

        optimizer.step(closure)
        return torch.stack(losses)

    x = torch.nn.Parameter(initial.clone())
    optimizer = optimizer_for(x)
    step(x, optimizer)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    resumed_x = torch.nn.Parameter(x.detach().clone())
    expected_losses = step(x, optimizer)

    resumed = optimizer_for(resumed_x)
    resumed.load_state_dict(checkpoint)
    actual_losses = step(resumed_x, resumed)
    torch.testing.assert_close(resumed_x, x, atol=0, rtol=0)
    torch.testing.assert_close(actual_losses, expected_losses, atol=0, rtol=0)
    for key, expected in optimizer.state[x].items():
        actual = resumed.state[resumed_x][key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert actual == expected
