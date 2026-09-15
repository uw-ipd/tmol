"""Correctness and derivative gates for live grid-scoring LK-ball terms."""

from __future__ import annotations

from dataclasses import replace

import attr
import pytest
import torch

from tmol.score import beta2016_score_function
from tmol.score.gridscore import (
    LK_BALL_SCORE_TYPES,
    build_live_lk_ball_data,
    score_ligand_lk_ball,
    score_ligand_lk_ball_with_gradients,
)
from tmol.score.gridscore.live_lk_ball import _live_lk_ball_op
from tmol.tests.ligand.test_ligand_receptor_perterm_baseline import (
    _GOLDEN_PER_TERM,
    _TARGETS,
    _TERM_ORDER,
)
from tmol.tests.ligand.test_protein_ligand_ddg import LIGAND_RES_NAME
from tmol.tests.score.gridscore.test_gridscore import _load_target_pose


def _ligand_block(pose_stack) -> int:
    matches = []
    for block_index, type_index_tensor in enumerate(pose_stack.block_type_ind64[0]):
        type_index = int(type_index_tensor)
        if type_index < 0:
            continue
        if (
            pose_stack.packed_block_types.active_block_types[type_index].name3
            == LIGAND_RES_NAME
        ):
            matches.append(block_index)
    if len(matches) != 1:
        raise ValueError(f"expected one ligand block, found {len(matches)}")
    return matches[0]


def _ligand_coords(pose_stack, ligand_block: int) -> torch.Tensor:
    type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    n_atoms = pose_stack.packed_block_types.active_block_types[type_index].n_atoms
    offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    return pose_stack.coords[:, offset : offset + n_atoms]


@pytest.mark.parametrize("target", _TARGETS)
def test_live_lk_ball_matches_fp64_interface_golden(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db)
    raw = score_ligand_lk_ball(data, _ligand_coords(pose_stack, ligand_block))[0]
    score_function = beta2016_score_function(torch.device("cpu"), param_db=param_db)
    weights = torch.stack(
        [score_function.get_weight(score_type) for score_type in LK_BALL_SCORE_TYPES]
    ).to(torch.float64)
    actual = raw * weights
    expected = torch.tensor(
        [
            _GOLDEN_PER_TERM[target]["interface"][_TERM_ORDER.index(score_type.name)]
            for score_type in LK_BALL_SCORE_TYPES
        ],
        dtype=torch.float64,
    )
    errors = torch.abs(actual - expected)
    for score_type, actual_term, expected_term, error in zip(
        LK_BALL_SCORE_TYPES, actual, expected, errors
    ):
        print(
            f"{target} live {score_type.name}: actual={float(actual_term):.12f} "
            f"expected={float(expected_term):.12f} abs_error={float(error):.3e}"
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_live_lk_ball_gradients_match_finite_difference(device):
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    coords = _ligand_coords(pose_stack, ligand_block).to(device).contiguous()
    score_function = beta2016_score_function(device, param_db=param_db)
    weights = torch.stack(
        [score_function.get_weight(score_type) for score_type in LK_BALL_SCORE_TYPES]
    ).to(device=device, dtype=torch.float64)
    energies, gradients = score_ligand_lk_ball_with_gradients(data, coords)
    assert torch.all(energies[0] != 0)

    step = 1.0e-6
    finite_by_channel = torch.empty_like(gradients)
    for column in range(coords.numel()):
        plus = coords.clone().reshape(-1)
        minus = coords.clone().reshape(-1)
        plus[column] += step
        minus[column] -= step
        finite_by_channel[0, :, column // 3, column % 3] = (
            score_ligand_lk_ball(data, plus.reshape_as(coords))[0]
            - score_ligand_lk_ball(data, minus.reshape_as(coords))[0]
        ) / (2.0 * step)

    differentiable = coords.clone().requires_grad_(True)
    torch.sum(score_ligand_lk_ball(data, differentiable) * weights).backward()
    analytic = torch.sum(gradients * weights[None, :, None, None], dim=1)
    finite = torch.sum(finite_by_channel * weights[None, :, None, None], dim=1)
    channel_errors = torch.amax(torch.abs(gradients - finite_by_channel), dim=(0, 2, 3))
    autograd_error = torch.max(torch.abs(analytic - differentiable.grad))
    for score_type, error in zip(LK_BALL_SCORE_TYPES, channel_errors):
        print(
            f"live {score_type.name} ligand-coordinate gradient: "
            f"max |analytic-finite|={float(error):.3e}"
        )
    print(f"live LK-ball max |analytic-autograd|={float(autograd_error):.3e}")
    torch.testing.assert_close(gradients, finite_by_channel, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(analytic, finite, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(analytic, differentiable.grad, rtol=0.0, atol=0.0)


def test_live_lk_ball_cell_list_matches_brute_force_exactly():
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db)
    base = _ligand_coords(pose_stack, ligand_block)
    shifts = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, -0.02, 0.01], [-0.04, 0.01, 0.02]],
        dtype=torch.float64,
    )
    coords = base + shifts[:, None, :]
    cell_energy, cell_gradient = _live_lk_ball_op(data, coords, use_cell_list=True)
    brute_energy, brute_gradient = _live_lk_ball_op(data, coords, use_cell_list=False)
    energy_error = torch.max(torch.abs(cell_energy - brute_energy))
    gradient_error = torch.max(torch.abs(cell_gradient - brute_gradient))
    print(
        "live LK-ball cell-list/brute-force: "
        f"max energy error={float(energy_error):.3e} "
        f"max gradient error={float(gradient_error):.3e}"
    )
    torch.testing.assert_close(cell_energy, brute_energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cell_gradient, brute_gradient, rtol=0.0, atol=0.0)


def test_live_lk_ball_rejects_invalid_cpu_topology():
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db)
    active = torch.nonzero(data.ligand_water_kinds != 0)
    assert active.shape[0] > 0
    polar_site, water = active[0].tolist()
    invalid_atoms = data.ligand_water_atoms.clone()
    invalid_atoms[polar_site, water, 0] = data.n_ligand_atoms
    with pytest.raises(ValueError, match="out-of-range"):
        replace(data, ligand_water_atoms=invalid_atoms)


def test_live_lk_ball_rejects_covalently_connected_ligand():
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    receptor_block = next(
        block
        for block in range(pose_stack.max_n_blocks)
        if block != ligand_block and int(pose_stack.block_type_ind64[0, block]) >= 0
    )
    connections = pose_stack.inter_residue_connections64.clone()
    connections[0, ligand_block, 0] = torch.tensor((receptor_block, 0))
    connected_pose = attr.evolve(pose_stack, inter_residue_connections64=connections)
    with pytest.raises(ValueError, match="no covalent receptor connections"):
        build_live_lk_ball_data(connected_pose, ligand_block, param_db)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_live_lk_ball_kernel_handles_mutated_data_without_oob(device):
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    coords = _ligand_coords(pose_stack, ligand_block).to(device)
    active = torch.nonzero(data.ligand_water_kinds != 0)
    polar_site, water = active[0].tolist()
    data.ligand_water_atoms[polar_site, water, 0] = data.n_ligand_atoms
    assert torch.isnan(score_ligand_lk_ball(data, coords)).all()

    data = build_live_lk_ball_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    data.cell_origin[0] = torch.nan
    assert torch.isnan(score_ligand_lk_ball(data, coords)).all()


@pytest.mark.parametrize(
    "device,dtype,rtol,atol",
    [
        pytest.param(
            torch.device("cpu"), torch.float64, 1.0e-12, 1.0e-10, id="cpu_fp64"
        ),
        pytest.param(torch.device("cpu"), torch.float32, 1.0e-5, 5.0e-5, id="cpu_fp32"),
        pytest.param(
            torch.device("cuda"),
            torch.float64,
            1.0e-12,
            1.0e-10,
            id="cuda_fp64",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            1.0e-5,
            5.0e-5,
            id="cuda_fp32",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_live_lk_ball_batched_device_agreement_and_noncontiguous_input(
    device, dtype, rtol, atol
):
    pose_stack, param_db = _load_target_pose("fxa")
    ligand_block = _ligand_block(pose_stack)
    cpu_data = build_live_lk_ball_data(pose_stack, ligand_block, param_db)
    base = _ligand_coords(pose_stack, ligand_block)
    shifts = torch.tensor(
        [[0.0, 0.0, 0.0], [0.03, -0.02, 0.01], [-0.04, 0.01, 0.02]],
        dtype=torch.float64,
    )
    contiguous = base + shifts[:, None, :]
    storage = torch.empty((*contiguous.shape[:-1], 6), dtype=torch.float64)
    storage[..., ::2] = contiguous
    coords = storage[..., ::2]
    assert not coords.is_contiguous()
    expected, expected_gradient = score_ligand_lk_ball_with_gradients(cpu_data, coords)
    device_data = cpu_data.to(device, dtype)
    actual, actual_gradient = score_ligand_lk_ball_with_gradients(
        device_data, coords.to(device=device, dtype=dtype)
    )
    actual = actual.cpu().to(torch.float64)
    actual_gradient = actual_gradient.cpu().to(torch.float64)
    error = torch.max(torch.abs(actual - expected))
    gradient_error = torch.max(torch.abs(actual_gradient - expected_gradient))
    print(
        f"live LK-ball CPU/{device.type} batched: "
        f"max energy error={float(error):.3e} "
        f"max gradient error={float(gradient_error):.3e}"
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)

    differentiable = coords.to(device=device, dtype=dtype).requires_grad_(True)
    autograd_energy = score_ligand_lk_ball(device_data, differentiable)
    torch.sum(autograd_energy).backward()
    _, direct_gradient = score_ligand_lk_ball_with_gradients(
        device_data, differentiable.detach()
    )
    torch.testing.assert_close(
        differentiable.grad,
        torch.sum(direct_gradient, dim=1),
        rtol=0.0,
        atol=0.0,
    )
