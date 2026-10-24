"""Correctness and derivative gates for live grid-scoring hydrogen bonds."""

from __future__ import annotations

from dataclasses import replace

import numpy
import pytest
import torch

from tmol.score import beta2016_score_function
from tmol.score.gridscore import (
    build_live_hbond_data,
    score_ligand_hbond,
    score_ligand_hbond_with_gradients,
)
from tmol.score import ScoreType
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
def test_live_hbond_matches_fp64_interface_golden(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    data = build_live_hbond_data(pose_stack, ligand_block, param_db)
    coords = _ligand_coords(pose_stack, ligand_block)
    raw = score_ligand_hbond(data, coords)[0]
    weight = beta2016_score_function(torch.device("cpu"), param_db=param_db).get_weight(
        ScoreType.hbond
    )
    actual = raw * weight.to(torch.float64)
    expected = _GOLDEN_PER_TERM[target]["interface"][_TERM_ORDER.index("hbond")]
    error = abs(float(actual) - expected)
    print(
        f"{target} live hbond: actual={float(actual):.12f} "
        f"expected={expected:.12f} abs_error={error:.3e}"
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-6,
    )


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
def test_live_hbond_analytic_and_autograd_gradients_match_finite_difference(
    device,
):
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_hbond_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    coords = _ligand_coords(pose_stack, ligand_block).to(device).contiguous()
    energies, analytic = score_ligand_hbond_with_gradients(data, coords)
    assert energies[0] != 0

    step = 1.0e-6
    finite = torch.empty_like(coords)
    for column in range(coords.numel()):
        plus = coords.clone().reshape(-1)
        minus = coords.clone().reshape(-1)
        plus[column] += step
        minus[column] -= step
        finite.reshape(-1)[column] = (
            score_ligand_hbond(data, plus.reshape_as(coords))[0]
            - score_ligand_hbond(data, minus.reshape_as(coords))[0]
        ) / (2.0 * step)

    differentiable = coords.clone().requires_grad_(True)
    score_ligand_hbond(data, differentiable).sum().backward()
    finite_error = torch.max(torch.abs(analytic - finite))
    autograd_error = torch.max(torch.abs(analytic - differentiable.grad))
    print(
        "live hbond ligand-coordinate gradient: "
        f"max |analytic-finite|={float(finite_error):.3e} "
        f"max |analytic-autograd|={float(autograd_error):.3e}"
    )
    torch.testing.assert_close(analytic, finite, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(analytic, differentiable.grad, rtol=0.0, atol=0.0)


def _brute_force_reference(data, coords):
    from tmol.tests.score.hbond.potentials import hbond_score_V_dV

    cutoff = float(data.global_params[0, 5])
    total = 0.0
    coords = coords.detach().cpu().numpy()
    receptor_donors = data.receptor_donors.numpy()
    receptor_acceptors = data.receptor_acceptors.numpy()
    pair_params = data.pair_params.numpy()
    pair_polynomials = data.pair_polynomials.numpy()
    global_params = data.global_params[0].numpy()
    for ligand_site, donor_type in enumerate(data.ligand_donor_types.tolist()):
        donor_atom, hydrogen_atom = data.ligand_donor_atoms[ligand_site].tolist()
        donor = coords[donor_atom]
        hydrogen = coords[hydrogen_atom]
        for receptor_site, acceptor_type in enumerate(
            data.receptor_acceptor_types.tolist()
        ):
            acceptor, base, base0 = receptor_acceptors[receptor_site]
            if numpy.linalg.norm(hydrogen - acceptor) >= cutoff:
                continue
            total += hbond_score_V_dV(
                D=donor,
                H=hydrogen,
                A=acceptor,
                B=base,
                B0=base0,
                pair_params=pair_params[donor_type, acceptor_type],
                polynomials=pair_polynomials[donor_type, acceptor_type],
                global_params=global_params,
            )[0]
    for ligand_site, acceptor_type in enumerate(data.ligand_acceptor_types.tolist()):
        acceptor_atom, base_atom, base0_atom = data.ligand_acceptor_atoms[
            ligand_site
        ].tolist()
        acceptor = coords[acceptor_atom]
        base = coords[base_atom]
        base0 = coords[base0_atom]
        for receptor_site, donor_type in enumerate(data.receptor_donor_types.tolist()):
            donor, hydrogen = receptor_donors[receptor_site]
            if numpy.linalg.norm(hydrogen - acceptor) >= cutoff:
                continue
            total += hbond_score_V_dV(
                D=donor,
                H=hydrogen,
                A=acceptor,
                B=base,
                B0=base0,
                pair_params=pair_params[donor_type, acceptor_type],
                polynomials=pair_polynomials[donor_type, acceptor_type],
                global_params=global_params,
            )[0]
    return total


def _active_ligand_anchor(data, coords):
    from tmol.tests.score.hbond.potentials import hbond_score_V_dV

    cutoff = float(data.global_params[0, 5])
    xyz = coords.detach().cpu().numpy()
    globals_ = data.global_params[0].numpy()
    for ligand_site, donor_type in enumerate(data.ligand_donor_types.tolist()):
        donor_atom, hydrogen_atom = data.ligand_donor_atoms[ligand_site].tolist()
        donor = xyz[donor_atom]
        hydrogen = xyz[hydrogen_atom]
        for receptor_site, acceptor_type in enumerate(
            data.receptor_acceptor_types.tolist()
        ):
            acceptor, base, base0 = data.receptor_acceptors[receptor_site].numpy()
            if numpy.linalg.norm(hydrogen - acceptor) >= cutoff:
                continue
            value = hbond_score_V_dV(
                D=donor,
                H=hydrogen,
                A=acceptor,
                B=base,
                B0=base0,
                pair_params=data.pair_params[donor_type, acceptor_type].numpy(),
                polynomials=data.pair_polynomials[donor_type, acceptor_type].numpy(),
                global_params=globals_,
            )[0]
            if abs(value) > 1.0e-8:
                return hydrogen_atom
    for ligand_site, acceptor_type in enumerate(data.ligand_acceptor_types.tolist()):
        acceptor_atom, base_atom, base0_atom = data.ligand_acceptor_atoms[
            ligand_site
        ].tolist()
        acceptor = xyz[acceptor_atom]
        base = xyz[base_atom]
        base0 = xyz[base0_atom]
        for receptor_site, donor_type in enumerate(data.receptor_donor_types.tolist()):
            donor, hydrogen = data.receptor_donors[receptor_site].numpy()
            if numpy.linalg.norm(hydrogen - acceptor) >= cutoff:
                continue
            value = hbond_score_V_dV(
                D=donor,
                H=hydrogen,
                A=acceptor,
                B=base,
                B0=base0,
                pair_params=data.pair_params[donor_type, acceptor_type].numpy(),
                polynomials=data.pair_polynomials[donor_type, acceptor_type].numpy(),
                global_params=globals_,
            )[0]
            if abs(value) > 1.0e-8:
                return acceptor_atom
    raise AssertionError("cdk2 must contain a contributing interface hbond site")


def test_live_hbond_cell_list_matches_brute_force_across_cell_boundary():
    from tmol.score.gridscore._common import build_cell_list as _build_cell_list

    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_hbond_data(pose_stack, ligand_block, param_db)
    base = _ligand_coords(pose_stack, ligand_block)[0]
    anchor_atom = _active_ligand_anchor(data, base)
    cutoff = float(data.global_params[0, 5])
    receptor_anchors = torch.cat(
        (data.receptor_donors[:, 1], data.receptor_acceptors[:, 0])
    )
    origin = torch.floor(receptor_anchors.min(dim=0).values / cutoff) * cutoff
    anchor_x = float(base[anchor_atom, 0])
    minimum_x = float(receptor_anchors[:, 0].min())
    boundary_index = max(1, int(numpy.ceil((anchor_x - minimum_x) / cutoff)))
    origin[0] = anchor_x - boundary_index * cutoff
    cell_coords = torch.floor((receptor_anchors - origin) / cutoff).to(torch.int64)
    dims = cell_coords.max(dim=0).values + 1
    donor_offsets, donor_indices = _build_cell_list(
        data.receptor_donors[:, 1], origin, dims, cutoff
    )
    acceptor_offsets, acceptor_indices = _build_cell_list(
        data.receptor_acceptors[:, 0], origin, dims, cutoff
    )
    data = replace(
        data,
        cell_origin=origin.contiguous(),
        cell_dims=dims.contiguous(),
        receptor_donor_cell_offsets=donor_offsets,
        receptor_donor_cell_indices=donor_indices,
        receptor_acceptor_cell_offsets=acceptor_offsets,
        receptor_acceptor_cell_indices=acceptor_indices,
    )
    epsilon = 1.0e-6
    shifts = torch.tensor(
        [
            [-epsilon, 0.0, 0.0],
            [epsilon, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    coords = base[None, :, :] + shifts[:, None, :]
    anchor_cells = torch.floor(
        (coords[:, anchor_atom, 0] - data.cell_origin[0]) / cutoff
    ).to(torch.int64)
    assert anchor_cells.tolist() == [boundary_index - 1, boundary_index]
    actual = score_ligand_hbond(data, coords)
    expected = torch.tensor(
        [_brute_force_reference(data, pose) for pose in coords],
        dtype=torch.float64,
    )
    assert torch.all(expected != 0)
    error = torch.max(torch.abs(actual - expected))
    print(f"live hbond cell-list/brute-force max error={float(error):.3e}")
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-10)


def test_live_hbond_rejects_invalid_cpu_topology():
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_hbond_data(pose_stack, ligand_block, param_db)
    if data.ligand_donor_atoms.shape[0]:
        invalid_atoms = data.ligand_donor_atoms.clone()
        invalid_atoms[0, 0] = data.n_ligand_atoms
        with pytest.raises(ValueError, match="out-of-range"):
            replace(data, ligand_donor_atoms=invalid_atoms)
    else:
        invalid_atoms = data.ligand_acceptor_atoms.clone()
        invalid_atoms[0, 0] = data.n_ligand_atoms
        with pytest.raises(ValueError, match="out-of-range"):
            replace(data, ligand_acceptor_atoms=invalid_atoms)


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
def test_live_hbond_kernel_handles_mutated_device_data_without_oob(device):
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    data = build_live_hbond_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    coords = _ligand_coords(pose_stack, ligand_block).to(device)

    if data.ligand_donor_atoms.shape[0]:
        data.ligand_donor_atoms[0, 0] = data.n_ligand_atoms
    else:
        data.ligand_acceptor_atoms[0, 0] = data.n_ligand_atoms
    assert torch.isnan(score_ligand_hbond(data, coords)).all()

    data = build_live_hbond_data(pose_stack, ligand_block, param_db).to(
        device, torch.float64
    )
    data.cell_origin[0] = torch.nan
    assert torch.isnan(score_ligand_hbond(data, coords)).all()


@pytest.mark.parametrize(
    "device,dtype,rtol,atol",
    [
        pytest.param(
            torch.device("cpu"),
            torch.float64,
            1.0e-12,
            1.0e-10,
            id="cpu_fp64",
        ),
        pytest.param(
            torch.device("cpu"),
            torch.float32,
            1.0e-5,
            5.0e-5,
            id="cpu_fp32",
        ),
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
def test_live_hbond_batched_device_agreement_and_noncontiguous_input(
    device, dtype, rtol, atol
):
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    cpu_data = build_live_hbond_data(pose_stack, ligand_block, param_db)
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
    expected = score_ligand_hbond(cpu_data, coords)
    actual = (
        score_ligand_hbond(
            cpu_data.to(device, dtype), coords.to(device=device, dtype=dtype)
        )
        .cpu()
        .to(torch.float64)
    )
    error = torch.max(torch.abs(actual - expected))
    print(f"live hbond CPU/{device.type} batched max error={float(error):.3e}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
