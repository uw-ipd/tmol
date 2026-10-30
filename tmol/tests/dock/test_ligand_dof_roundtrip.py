"""Phase-1 gates for Rosetta-faithful ligand docking DOFs."""

from __future__ import annotations


import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.io._pose_stack_from_biotite import pose_stack_from_atom_array
from tmol.optimization._minimizers import _build_kinforest_network
from tmol.score import ScoreFunction
from tmol.tests.data import data_path
from tmol.tests.ligand.test_protein_ligand_ddg import (
    LIGAND_RES_NAME,
    _load_complex_cif,
)

PLI_DATA_DIR = data_path("protein_ligand_test")


def _load_cdk2_pose():
    """Use exactly the cdk2 construction path from the gridscore oracle."""

    param_db = ParameterDatabase.get_default()
    pose_stack, context = pose_stack_from_atom_array(
        _load_complex_cif("cdk2"),
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / "cdk2.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=param_db,
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=torch.float64)
    return pose_stack, context.parameter_database


@pytest.fixture(scope="module")
def cdk2_pose_and_db():
    return _load_cdk2_pose()


def _ligand_block(pose_stack) -> int:
    found = []
    for block in range(pose_stack.max_n_blocks):
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        if block_type_index < 0:
            continue
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        if block_type.name3 == LIGAND_RES_NAME:
            found.append(block)
    assert len(found) == 1
    return found[0]


def _empty_score_function(param_db, device):
    # Kinematic round trips need the production network surface but no energies.
    return ScoreFunction(param_db, device)


def _network(pose_stack, param_db, docking_dofs, kin_dtype=torch.float64):
    return _build_kinforest_network(
        pose_stack,
        _empty_score_function(param_db, pose_stack.device),
        docking_dofs.fold_forest,
        docking_dofs.move_map,
        kin_dtype=kin_dtype,
    )


@pytest.mark.parametrize(
    ("coord_dtype", "kin_dtype", "bound"),
    [
        (torch.float64, torch.float64, 1.0e-9),
        # fp32 coords through inverse+forward kinematics accumulate ~1e-4 A
        # (measured 1.09e-4 on cdk2); this loose bound catches gross errors while
        # tolerating fp32 kinematic round-off. The exact value is printed.
        (torch.float32, torch.float32, 5.0e-4),
    ],
)
def test_cdk2_coords_dofs_coords_roundtrip(
    cdk2_pose_and_db, coord_dtype, kin_dtype, bound
):
    original, param_db = cdk2_pose_and_db
    pose_stack = original.clone()
    pose_stack.coords = pose_stack.coords.to(coord_dtype)
    ligand_block = _ligand_block(pose_stack)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    network = _network(pose_stack, param_db, docking_dofs, kin_dtype=kin_dtype)
    rebuilt = network.pose_stack_from_dofs()
    error = float(torch.max(torch.abs(rebuilt.coords - pose_stack.coords)))

    print(f"cdk2 {coord_dtype} coords->DOFs->coords max_abs_error={error:.9e}")
    assert error < bound


def _load_cdk2_pose_on_device(device: torch.device):
    if device.type == "cpu":
        return _load_cdk2_pose()

    # The Phase-1 helper is CPU/fp64-only. Repeat its exact construction on the
    # requested device for the GPU/fp32 batched lane.
    pose_stack, context = pose_stack_from_atom_array(
        _load_complex_cif("cdk2"),
        device,
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / "cdk2.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=torch.float32)
    return pose_stack, context.parameter_database
