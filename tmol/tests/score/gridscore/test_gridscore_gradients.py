"""M2e hard gates for analytic grid-coordinate derivatives."""

from __future__ import annotations

import pytest
import torch

from tmol.score.gridscore import GridSet, score_ligand_grid_batched
from tmol.score.gridscore.grid import (
    GridGeometry,
    interpolate_grid,
    score_ligand_grid,
    score_ligand_grid_batched_with_gradients,
    xform_rep,
)
from tmol.score.gridscore.grid import _interpolate_grid_and_penalty


def _finite_difference(function, coordinates, step=1.0e-6):
    result = torch.empty_like(coordinates)
    flat_result = result.reshape(-1)
    for column in range(coordinates.numel()):
        plus = coordinates.detach().clone().reshape(-1)
        minus = coordinates.detach().clone().reshape(-1)
        plus[column] += step
        minus[column] -= step
        flat_result[column] = (
            function(plus.reshape_as(coordinates))
            - function(minus.reshape_as(coordinates))
        ) / (2.0 * step)
    return result


def _gradient_grid(device):
    dtype = torch.float64
    dims = (6, 7, 8)
    x, y, z = torch.meshgrid(
        *(torch.arange(n, dtype=dtype) for n in dims), indexing="ij"
    )
    typed = []
    raw_rep = []
    sol = []
    for grid in range(3):
        phase = float(grid + 1)
        field = (
            torch.sin((x + phase) * 0.31)
            + torch.cos((y - phase) * 0.23)
            + 0.07 * z * z
            + 0.03 * x * y
        )
        typed.append(field)
        raw_rep.append(0.2 + (0.15 * field + 0.1 * phase) ** 2)
        sol.append(0.4 * field - 0.02 * y * z)
    elec = 0.2 * torch.sin(0.17 * x * y) - 0.11 * z + 0.03 * x
    geometry = GridGeometry(
        origin=torch.tensor([-1.0, 2.0, -3.0], dtype=dtype),
        dims=dims,
        voxel_spacing=0.5,
        ligand_com=torch.zeros(3, dtype=dtype),
        max_radius=7.5,
    )
    return GridSet(
        geometry=geometry,
        atom_type_names=("A", "B", "C"),
        atom_type_indices=torch.tensor([1, 3, 7], dtype=torch.int64),
        fa_atr=torch.stack(typed),
        fa_rep=xform_rep(torch.stack(raw_rep)),
        fa_sol=torch.stack(sol),
        fa_elec=elec,
    ).to(device, dtype=dtype)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu_fp64"),
        pytest.param(
            torch.device("cuda"),
            id="cuda_fp64",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_compiled_spatial_derivative_matches_finite_difference_and_autograd(device):
    grid = _gradient_grid(device)
    coefficients = grid.fa_atr_coeffs
    indices = torch.tensor(
        [
            [2.9997, 2.25, 3.10],  # near an interior voxel boundary
            [0.0004, 4.20, 1.80],  # near the lower grid edge
            [5.20, 5.10, 6.30],  # upper-x OOB penalty region
            [-0.15, 1.70, 7.25],  # lower-x and upper-z OOB region
        ],
        dtype=torch.float64,
        device=device,
    )
    grid_indices = torch.tensor([0, 1, 2, 1], dtype=torch.int64, device=device)

    from tmol.score.gridscore.potentials.compiled import (
        interpolate_with_derivatives,
    )

    values, analytic, penalties, analytic_penalty = interpolate_with_derivatives(
        coefficients.contiguous(),
        indices.contiguous(),
        grid_indices,
        50.0,
    )
    differentiable_indices = indices.detach().clone().requires_grad_(True)
    autograd_values, autograd_penalties = _interpolate_grid_and_penalty(
        coefficients, differentiable_indices, grid_indices
    )
    value_grad = torch.autograd.grad(
        autograd_values.sum(), differentiable_indices, retain_graph=True
    )[0]
    total_grad = torch.autograd.grad(
        (autograd_values + autograd_penalties).sum(), differentiable_indices
    )[0]

    finite_value = _finite_difference(
        lambda query: interpolate_grid(coefficients, query, grid_indices).sum(),
        indices,
    )
    finite_total = _finite_difference(
        lambda query: sum(
            tensor.sum()
            for tensor in _interpolate_grid_and_penalty(
                coefficients, query, grid_indices
            )
        ),
        indices,
    )
    value_error = torch.max(torch.abs(analytic - finite_value))
    total_error = torch.max(torch.abs(analytic + analytic_penalty - finite_total))
    autograd_error = torch.max(torch.abs(value_grad - analytic))
    autograd_total_error = torch.max(
        torch.abs(total_grad - analytic - analytic_penalty)
    )
    print(
        f"{device} compiled grid gradient max errors: "
        f"value-vs-FD={float(value_error):.3e}, "
        f"total-vs-FD={float(total_error):.3e}, "
        f"autograd-value={float(autograd_error):.3e}, "
        f"autograd-total={float(autograd_total_error):.3e}"
    )
    torch.testing.assert_close(values, autograd_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(penalties, autograd_penalties, rtol=0.0, atol=0.0)
    torch.testing.assert_close(analytic, finite_value, rtol=1.0e-8, atol=2.0e-7)
    torch.testing.assert_close(
        analytic + analytic_penalty,
        finite_total,
        rtol=1.0e-8,
        atol=2.0e-7,
    )
    torch.testing.assert_close(value_grad, analytic, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        total_grad, analytic + analytic_penalty, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu_fp64"),
        pytest.param(
            torch.device("cuda"),
            id="cuda_fp64",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_grid_interface_coordinate_gradient_gate_per_channel_and_total(device):
    grid = _gradient_grid(device)
    fractional_indices = torch.tensor(
        [
            [2.9997, 2.25, 3.10],
            [0.0004, 4.20, 1.80],
            [5.20, 5.10, 6.30],
            [-0.15, 1.70, 7.25],
        ],
        dtype=torch.float64,
        device=device,
    )
    coordinates = (
        grid.geometry.origin + grid.geometry.voxel_spacing * fractional_indices
    ).detach()
    atom_types = ("A", "B", "C", "B")
    charges = torch.tensor([0.25, -0.4, 0.7, -0.15], dtype=torch.float64, device=device)

    differentiable = coordinates.clone().requires_grad_(True)
    channels = score_ligand_grid(grid, differentiable, atom_types, charges)
    channel_errors = []
    for channel in range(4):
        analytic = torch.autograd.grad(
            channels[channel], differentiable, retain_graph=True
        )[0]
        finite = _finite_difference(
            lambda query, channel=channel: score_ligand_grid(
                grid, query, atom_types, charges
            )[channel],
            coordinates,
        )
        channel_errors.append(torch.max(torch.abs(analytic - finite)))

    differentiable_total = coordinates.clone().requires_grad_(True)
    total = score_ligand_grid_batched(
        grid,
        differentiable_total.unsqueeze(0),
        atom_types,
        charges,
    )[0]
    analytic_total = torch.autograd.grad(total, differentiable_total)[0]
    direct_total, direct_gradient = score_ligand_grid_batched_with_gradients(
        grid,
        differentiable_total.unsqueeze(0),
        atom_types,
        charges,
    )
    finite_total = _finite_difference(
        lambda query: score_ligand_grid_batched(
            grid, query.unsqueeze(0), atom_types, charges
        )[0],
        coordinates,
    )
    total_error = torch.max(torch.abs(analytic_total - finite_total))
    print(
        f"{device} grid interface gradient max errors: "
        f"atr={float(channel_errors[0]):.3e}, "
        f"rep={float(channel_errors[1]):.3e}, "
        f"sol={float(channel_errors[2]):.3e}, "
        f"elec={float(channel_errors[3]):.3e}, "
        f"total+OOB={float(total_error):.3e}"
    )
    for error in channel_errors:
        assert float(error) < 2.0e-7
    assert float(total_error) < 5.0e-7
    torch.testing.assert_close(direct_total[0], total, rtol=0.0, atol=2.0e-12)
    torch.testing.assert_close(
        direct_gradient[0], analytic_total, rtol=0.0, atol=2.0e-12
    )
    assert direct_total.grad_fn is None
    assert direct_gradient.grad_fn is None


def test_grid_edge_transition_is_c1_cpu_fp64():
    grid = _gradient_grid(torch.device("cpu"))
    coefficients = grid.fa_atr_coeffs
    grid_indices = torch.zeros(1, dtype=torch.int64)
    epsilon = 1.0e-7
    slopes = []
    for x_index in (-epsilon, 0.0, epsilon):
        query = torch.tensor(
            [[x_index, 2.2, 3.4]], dtype=torch.float64, requires_grad=True
        )
        values, penalties = _interpolate_grid_and_penalty(
            coefficients, query, grid_indices
        )
        slopes.append(torch.autograd.grad((values + penalties).sum(), query)[0][0, 0])
    # The mirror spline has zero endpoint slope; the OOB quadratic also has
    # zero slope at zero overshoot. Both one-sided derivatives converge to it.
    assert abs(float(slopes[1])) < 1.0e-12
    assert abs(float(slopes[0] - slopes[1])) < 2.0e-4
    assert abs(float(slopes[2] - slopes[1])) < 2.0e-4
