"""M2e hard gates for analytic grid-coordinate derivatives."""

from __future__ import annotations

import torch

from tmol.score.gridscore import GridSet
from tmol.score.gridscore.grid import GridGeometry, xform_rep
from tmol.score.gridscore.scorer import interpolate_grid


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
        values, penalties = interpolate_grid(
            coefficients, query, grid_indices, penalty=True
        )
        slopes.append(torch.autograd.grad((values + penalties).sum(), query)[0][0, 0])
    # The mirror spline has zero endpoint slope; the OOB quadratic also has
    # zero slope at zero overshoot. Both one-sided derivatives converge to it.
    assert abs(float(slopes[1])) < 1.0e-12
    assert abs(float(slopes[0] - slopes[1])) < 2.0e-4
    assert abs(float(slopes[2] - slopes[1])) < 2.0e-4
