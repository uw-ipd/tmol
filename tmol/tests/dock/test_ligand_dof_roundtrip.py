"""Phase-1 gates for Rosetta-faithful ligand docking DOFs."""

from __future__ import annotations


import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.io import pose_stack_from_biotite
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.optimization import build_kinforest_network
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
    pose_stack, context = pose_stack_from_biotite(
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
    return build_kinforest_network(
        pose_stack,
        _empty_score_function(param_db, pose_stack.device),
        docking_dofs.fold_forest,
        docking_dofs.move_map,
        kin_dtype=kin_dtype,
    )


def _ligand_kfo_nodes(network, pose_stack, ligand_block, pose=0):
    start = pose * pose_stack.max_n_pose_atoms + int(
        pose_stack.block_coord_offset64[pose, ligand_block]
    )
    block_type_index = int(pose_stack.block_type_ind64[pose, ligand_block])
    n_atoms = pose_stack.packed_block_types.active_block_types[block_type_index].n_atoms
    return torch.nonzero(
        (network.id >= start) & (network.id < start + n_atoms), as_tuple=False
    ).flatten()


def _torsion_values(kin_coords, nodes_by_local_atom, block_type):
    values = []
    for torsion_index in range(block_type.n_torsions):
        atoms = tuple(
            int(uaid[0]) for uaid in block_type.ordered_torsions[torsion_index]
        )
        selected = [
            kin_coords[nodes_by_local_atom[atom]].reshape(1, 3) for atom in atoms
        ]
        ba = selected[0] - selected[1]
        bc = selected[2] - selected[1]
        cd = selected[3] - selected[2]
        unit_bc = bc / torch.linalg.vector_norm(bc, dim=1, keepdim=True)
        v = ba - torch.sum(ba * unit_bc, dim=1, keepdim=True) * unit_bc
        w = cd - torch.sum(cd * unit_bc, dim=1, keepdim=True) * unit_bc
        x = torch.sum(v * w, dim=1)
        y = torch.sum(torch.linalg.cross(unit_bc, v) * w, dim=1)
        values.append(torch.atan2(y, x)[0])
    return torch.stack(values)


def _angle_delta(after, before):
    delta = after - before
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def test_cdk2_dof_count_and_bidirectional_mapping(cdk2_pose_and_db):
    pose_stack, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    kin_module = PoseStackKinematicsModule(pose_stack, docking_dofs.fold_forest)
    minmap = MinimizerMap(pose_stack, kin_module.kmd, docking_dofs.move_map)

    assert len(docking_dofs.torsions) == 2
    assert tuple(docking_dofs.logical_chi_to_raw_dof) == ("chi1", "chi2")
    assert int(torch.sum(minmap.dof_mask)) == 8
    assert [
        (mapping.name, mapping.raw_atom_name, mapping.dof_name, mapping.sign)
        for mapping in docking_dofs.torsions
    ] == [
        ("chi1", "C7", "phi_c", 1),
        ("chi2", "C7", "phi_p", -1),
    ]
    assert len(
        {(mapping.raw_atom, mapping.dof_type) for mapping in docking_dofs.torsions}
    ) == len(docking_dofs.torsions)

    network = _network(pose_stack, param_db, docking_dofs)
    ligand_nodes = _ligand_kfo_nodes(network, pose_stack, ligand_block)
    target_to_kfo = {
        int(network.id[node])
        - int(pose_stack.block_coord_offset64[0, ligand_block]): int(node)
        for node in ligand_nodes
    }
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    logical = _torsion_values(
        network.kin_module(network.full_dofs), target_to_kfo, block_type
    )
    for mapping in docking_dofs.torsions:
        raw = network.full_dofs[
            target_to_kfo[mapping.raw_atom], mapping.dof_type
        ].double()
        predicted = mapping.logical_from_raw(raw)[0]
        error = torch.abs(_angle_delta(predicted, logical[mapping.torsion_index]))
        assert float(error) < 1.0e-10


def test_logical_raw_conversion_is_inverse_across_wrap_boundary(
    cdk2_pose_and_db,
):
    pose_stack, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    epsilon = 1.0e-8
    boundary_values = torch.tensor(
        [torch.pi - epsilon, -torch.pi + epsilon], dtype=torch.float64
    )

    for mapping in docking_dofs.torsions:
        raw = mapping.raw_from_logical(boundary_values)
        logical_roundtrip = mapping.logical_from_raw(raw)
        torch.testing.assert_close(
            _angle_delta(logical_roundtrip, boundary_values),
            torch.zeros_like(boundary_values),
            rtol=0.0,
            atol=1.0e-12,
        )

        logical = mapping.logical_from_raw(boundary_values)
        raw_roundtrip = mapping.raw_from_logical(logical)
        torch.testing.assert_close(
            _angle_delta(raw_roundtrip, boundary_values),
            torch.zeros_like(boundary_values),
            rtol=0.0,
            atol=1.0e-12,
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
    pose_stack, context = pose_stack_from_biotite(
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
