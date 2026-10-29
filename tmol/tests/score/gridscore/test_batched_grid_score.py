from __future__ import annotations

import pytest
import torch

from tmol.score.gridscore import GridScorer, build_grid
from tmol.score.gridscore.grid import probe_atom_types_from_name3
from tmol.tests.ligand.test_protein_ligand_ddg import LIGAND_RES_NAME
from tmol.tests.score.gridscore.test_gridscore import (
    _ligand_scoring_inputs,
    _load_cdk2_pose,
)


@pytest.fixture(scope="module")
def cdk2_grid_and_ligand():
    pose_stack, param_db = _load_cdk2_pose()
    probe_names = probe_atom_types_from_name3(pose_stack, LIGAND_RES_NAME)
    grids = build_grid(
        pose_stack,
        probe_names,
        param_db,
        ligand_name3=LIGAND_RES_NAME,
    )
    return grids, _ligand_scoring_inputs(pose_stack)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    (
        (torch.float32, 2.0e-6, 2.0e-5),
        (torch.float64, 1.0e-12, 1.0e-10),
    ),
)
def test_batched_grid_score_matches_individual_cdk2_poses(
    cdk2_grid_and_ligand, torch_device, dtype, rtol, atol
):
    cpu_grids, ligand_inputs = cdk2_grid_and_ligand
    base_coords, atom_types, base_charges = ligand_inputs
    grids = cpu_grids.to(torch_device, dtype=dtype)
    base_coords = base_coords.to(device=torch_device, dtype=dtype)
    charges = base_charges.to(device=torch_device, dtype=dtype)
    shifts = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, -0.1, 0.15], [-0.125, 0.075, -0.05]],
        device=torch_device,
        dtype=dtype,
    )
    contiguous_pose_coords = base_coords.unsqueeze(0) + shifts.unsqueeze(1)
    coordinate_storage = torch.empty(
        (*contiguous_pose_coords.shape[:-1], 6),
        device=torch_device,
        dtype=dtype,
    )
    coordinate_storage[..., ::2] = contiguous_pose_coords
    pose_coords = coordinate_storage[..., ::2]
    assert not pose_coords.is_contiguous()

    scorer = GridScorer(grids, atom_types, charges, device=torch_device, dtype=dtype)
    actual = scorer(pose_coords)
    expected = torch.stack(
        [scorer.grid_channels(coords[None]).sum() for coords in pose_coords]
    )

    assert actual.shape == (pose_coords.shape[0],)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
