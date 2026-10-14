"""Phase-1 gates for Rosetta-faithful ligand docking DOFs."""

from __future__ import annotations

import copy
import inspect

import attr
import numpy
import pytest
import torch
from frozendict import frozendict

import tmol.dock.ligand_dofs as ligand_dofs_module
from tmol.database import ParameterDatabase
from tmol.dock import build_ligand_docking_dofs
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.move_map import MinimizerMap, MoveMap
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.optimization import build_kinforest_network
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
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
        ligand_params_files=[
            str(PLI_DATA_DIR / "cdk2.xtal-lig.mmff94.tmol")
        ],
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
        block_type = pose_stack.packed_block_types.active_block_types[
            block_type_index
        ]
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


def _raw_dof_column(network, pose_stack, ligand_block, mapping, pose=0) -> int:
    target_atom = (
        pose * pose_stack.max_n_pose_atoms
        + int(pose_stack.block_coord_offset64[pose, ligand_block])
        + mapping.raw_atom
    )
    nodes = torch.nonzero(network.id == target_atom, as_tuple=False).flatten()
    assert nodes.numel() == 1
    flat_dof = int(nodes[0]) * network.full_dofs.shape[1] + int(mapping.dof_type)
    columns = torch.nonzero(
        network._dof_flat_idx == flat_dof, as_tuple=False
    ).flatten()
    assert columns.numel() == 1
    return int(columns[0])


def _ligand_kfo_nodes(network, pose_stack, ligand_block, pose=0):
    start = (
        pose * pose_stack.max_n_pose_atoms
        + int(pose_stack.block_coord_offset64[pose, ligand_block])
    )
    block_type_index = int(pose_stack.block_type_ind64[pose, ligand_block])
    n_atoms = pose_stack.packed_block_types.active_block_types[
        block_type_index
    ].n_atoms
    return torch.nonzero(
        (network.id >= start) & (network.id < start + n_atoms), as_tuple=False
    ).flatten()


def _coords_from_masked_dofs(network, masked_dofs):
    full_flat = network.full_dofs.detach().reshape(-1).scatter(
        0, network._dof_flat_idx, masked_dofs
    )
    return network.kin_module(full_flat.view_as(network.full_dofs))


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


def _batched_torsion_values(
    network, pose_stack, ligand_block, full_dofs=None
):
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    if full_dofs is None:
        full_dofs = network.full_dofs
    kin_coords = network.kin_module(full_dofs)
    per_pose = []
    for pose in range(pose_stack.n_poses):
        ligand_nodes = _ligand_kfo_nodes(
            network, pose_stack, ligand_block, pose
        )
        block_offset = int(
            pose_stack.block_coord_offset64[pose, ligand_block]
        )
        pose_base = pose * pose_stack.max_n_pose_atoms
        target_to_kfo = {
            int(network.id[node]) - pose_base - block_offset: int(node)
            for node in ligand_nodes
        }
        per_pose.append(
            _torsion_values(kin_coords, target_to_kfo, block_type)
        )
    return torch.stack(per_pose)


def _angle_delta(after, before):
    delta = after - before
    return torch.atan2(torch.sin(delta), torch.cos(delta))


@pytest.mark.parametrize(
    ("response_matrix", "message"),
    [
        (((0.05, 0.0), (0.0, 1.0)), "unit magnitude"),
        # Isolated but non-unit self-slope: the affine mapping stores no scale
        # term, so a 0.5 response would silently miscalibrate the logical chi.
        (((0.5, 0.0), (0.0, 1.0)), "unit magnitude"),
        (((1.0, 0.02), (0.0, 1.0)), "changes another declared dihedral"),
        (((float("nan"), 0.0), (0.0, 1.0)), "finite declared-dihedral"),
    ],
)
def test_empirical_torsion_sign_guards_raise(
    monkeypatch, response_matrix, message
):
    response_matrix = torch.tensor(response_matrix, dtype=torch.float64)

    def synthetic_dihedrals(raw_dofs, atom_nodes):
        del atom_nodes
        controllers = raw_dofs[:2, 0]
        return (controllers @ response_matrix).reshape(1, 2)

    monkeypatch.setattr(
        ligand_dofs_module,
        "_batched_coord_dihedrals",
        synthetic_dihedrals,
    )
    raw_dofs = torch.zeros((2, 1), dtype=torch.float64)
    ligand_nodes = torch.tensor([[0, 1]], dtype=torch.int64)
    torsion_atoms = torch.zeros((2, 4), dtype=torch.int64)
    raw_nodes = torch.tensor([[0, 1]], dtype=torch.int64)
    dof_types = torch.zeros(2, dtype=torch.int64)

    with pytest.raises(ValueError, match=message):
        ligand_dofs_module._empirical_torsion_signs(
            lambda dofs: dofs,
            raw_dofs,
            ligand_nodes,
            torsion_atoms,
            raw_nodes,
            dof_types,
        )


def test_empirical_torsion_signs_measures_in_float64(monkeypatch):
    """Lock the fp32->fp64 upcast: float32 raw DOFs must be validated in float64.

    Removing ``measure_dofs = raw_dofs.double()`` would leave this test failing
    even though the fp64-input synthetic guards stay green, guarding against a
    silent regression of the ache/egfr fix (FINDINGS F47).
    """

    seen_dtypes = []

    def spy_kinematics(dofs):
        seen_dtypes.append(dofs.dtype)
        return dofs

    def unit_dihedrals(coords, atom_nodes):
        del atom_nodes
        return coords[:2, 0].reshape(1, 2)

    monkeypatch.setattr(
        ligand_dofs_module, "_batched_coord_dihedrals", unit_dihedrals
    )
    raw_dofs = torch.zeros((2, 1), dtype=torch.float32)
    signs = ligand_dofs_module._empirical_torsion_signs(
        spy_kinematics,
        raw_dofs,
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.zeros((2, 4), dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
    )
    assert seen_dtypes, "validation kinematics was never invoked"
    assert all(dtype == torch.float64 for dtype in seen_dtypes)
    assert signs.tolist() == [1, 1]


def test_empirical_torsion_signs_supports_no_declared_torsions():
    def unexpected_forward_kinematics(_):
        raise AssertionError("zero-torsion validation should not run kinematics")

    signs = ligand_dofs_module._empirical_torsion_signs(
        unexpected_forward_kinematics,
        torch.zeros((2, 1), dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.int64),
        torch.empty((0, 4), dtype=torch.int64),
        torch.empty((1, 0), dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
    )
    assert signs.dtype == torch.int64
    assert signs.shape == (0,)


def _scan_cache_values(owner):
    if not hasattr(owner, "gen_seg_scan_path_segs"):
        return None
    cache = owner.gen_seg_scan_path_segs
    values = {}
    for field in cache.__attrs_attrs__:
        value = getattr(cache, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = value.detach().clone()
        elif isinstance(value, numpy.ndarray):
            values[field.name] = value.copy()
        else:
            values[field.name] = value
    return values


def _assert_scan_cache_values_equal(owner, expected):
    if expected is None:
        return
    actual = _scan_cache_values(owner)
    assert actual is not None
    assert actual.keys() == expected.keys()
    for name in expected:
        if isinstance(expected[name], torch.Tensor):
            torch.testing.assert_close(actual[name], expected[name])
        elif isinstance(expected[name], numpy.ndarray):
            numpy.testing.assert_array_equal(actual[name], expected[name])
        else:
            assert actual[name] == expected[name]


def _reverse_named_torsion_declaration(block_type, torsion_name):
    torsion_index = next(
        i for i, torsion in enumerate(block_type.torsions)
        if torsion.name == torsion_name
    )
    torsion = block_type.torsions[torsion_index]
    reversed_torsion = attr.evolve(
        torsion, a=torsion.d, b=torsion.c, c=torsion.b, d=torsion.a
    )
    torsions = list(block_type.torsions)
    torsions[torsion_index] = reversed_torsion
    block_type.torsions = tuple(torsions)

    torsion_to_uaids = dict(block_type.torsion_to_uaids)
    torsion_to_uaids[torsion_name] = tuple(
        reversed(torsion_to_uaids[torsion_name])
    )
    block_type.torsion_to_uaids = frozendict(torsion_to_uaids)
    block_type.ordered_torsions = block_type.ordered_torsions.copy()
    block_type.ordered_torsions[torsion_index] = block_type.ordered_torsions[
        torsion_index, ::-1
    ]
    return torsion_index


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
        error = torch.abs(
            _angle_delta(predicted, logical[mapping.torsion_index])
        )
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


def test_reversed_torsion_declaration_uses_empirical_sign():
    pose_stack, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(pose_stack)
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    original_pbt = pose_stack.packed_block_types
    active_block_types = list(original_pbt.active_block_types)
    active_block_types[block_type_index] = copy.deepcopy(
        active_block_types[block_type_index]
    )
    pose_stack.packed_block_types = PackedBlockTypes.from_restype_list(
        original_pbt.chem_db,
        original_pbt.restype_set,
        active_block_types,
        original_pbt.device,
    )
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    torsion_index = _reverse_named_torsion_declaration(block_type, "chi1")
    pose_stack.packed_block_types.torsion_uaids[
        block_type_index, torsion_index
    ] = torch.flip(
        pose_stack.packed_block_types.torsion_uaids[
            block_type_index, torsion_index
        ],
        dims=(0,),
    )

    # Regenerate this fresh PBT's scan cache from the reversed declaration.
    if hasattr(pose_stack.packed_block_types, "gen_seg_scan_path_segs"):
        delattr(pose_stack.packed_block_types, "gen_seg_scan_path_segs")
    if hasattr(block_type, "gen_seg_scan_path_segs"):
        delattr(block_type, "gen_seg_scan_path_segs")

    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    mapping = docking_dofs.logical_chi_to_raw_dof["chi1"]
    assert (
        mapping.raw_atom_name,
        mapping.dof_name,
        mapping.sign,
    ) == ("C7", "phi_c", 1)

    network = _network(pose_stack, param_db, docking_dofs)
    ligand_nodes = _ligand_kfo_nodes(network, pose_stack, ligand_block)
    target_to_kfo = {
        int(network.id[node])
        - int(pose_stack.block_coord_offset64[0, ligand_block]): int(node)
        for node in ligand_nodes
    }
    base_dofs = network.masked_dofs.detach().double()
    column = _raw_dof_column(
        network, pose_stack, ligand_block, mapping
    )
    delta = 0.2
    perturbed = base_dofs.clone()
    perturbed[column] += delta
    changed_coords = _coords_from_masked_dofs(network, perturbed)
    true_dihedral = _torsion_values(
        changed_coords, target_to_kfo, block_type
    )[torsion_index]
    raw = network.full_dofs[
        target_to_kfo[mapping.raw_atom], mapping.dof_type
    ].double() + delta
    predicted = mapping.logical_from_raw(raw)[0]
    assert float(torch.abs(_angle_delta(predicted, true_dihedral))) < 1.0e-10


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


def test_cdk2_ligand_dofs_are_independent_and_isolated(cdk2_pose_and_db):
    pose_stack, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    network = _network(pose_stack, param_db, docking_dofs)
    assert network.masked_dofs.numel() == 6 + len(docking_dofs.torsions)

    ligand_nodes = _ligand_kfo_nodes(network, pose_stack, ligand_block)
    target_to_kfo = {
        int(network.id[node])
        - int(pose_stack.block_coord_offset64[0, ligand_block]): int(node)
        for node in ligand_nodes
    }
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]

    base_dofs = network.masked_dofs.detach().double()
    base_kin_coords = _coords_from_masked_dofs(network, base_dofs)
    base_torsions = _torsion_values(base_kin_coords, target_to_kfo, block_type)
    epsilon = 1.0e-5

    # Each logical-chi controller changes its own chi with the recorded sign,
    # and leaves every other logical chi stationary to first order.
    for intended, mapping in enumerate(docking_dofs.torsions):
        column = _raw_dof_column(
            network, pose_stack, ligand_block, mapping
        )
        perturbed = base_dofs.clone()
        perturbed[column] += epsilon
        changed_coords = _coords_from_masked_dofs(network, perturbed)
        changed_torsions = _torsion_values(
            changed_coords, target_to_kfo, block_type
        )
        derivative = _angle_delta(changed_torsions, base_torsions) / epsilon
        expected = torch.zeros_like(derivative)
        expected[intended] = mapping.sign
        torch.testing.assert_close(derivative, expected, rtol=1.0e-4, atol=1.0e-5)

    # All eight free DOFs move ligand coordinates only.
    ids = network.id[1:].to(torch.int64)
    ligand_target_start = int(
        pose_stack.block_coord_offset64[0, ligand_block]
    )
    ligand_target_end = ligand_target_start + block_type.n_atoms
    receptor_nodes = torch.nonzero(
        (ids < ligand_target_start) | (ids >= ligand_target_end),
        as_tuple=False,
    ).flatten() + 1
    for column in range(base_dofs.numel()):
        perturbed = base_dofs.clone()
        perturbed[column] += epsilon
        changed_coords = _coords_from_masked_dofs(network, perturbed)
        receptor_error = torch.max(
            torch.abs(changed_coords[receptor_nodes] - base_kin_coords[receptor_nodes])
        )
        assert float(receptor_error) < 1.0e-12
        assert float(
            torch.max(
                torch.abs(changed_coords[ligand_nodes] - base_kin_coords[ligand_nodes])
            )
        ) > 0.0

    # The coordinate Jacobian for all ligand atoms has independent columns for
    # the six rigid-body and two chi variables.
    def ligand_coords(masked_dofs):
        return _coords_from_masked_dofs(network, masked_dofs)[ligand_nodes].reshape(-1)

    jacobian = torch.autograd.functional.jacobian(ligand_coords, base_dofs)
    rank = int(torch.linalg.matrix_rank(jacobian))
    singular_values = torch.linalg.svdvals(jacobian)
    print(
        "cdk2 ligand-coordinate Jacobian singular values:",
        singular_values.detach().cpu().tolist(),
    )
    assert rank == 6 + len(docking_dofs.torsions)


def test_cdk2_stacked_pose_dof_wiring_and_roundtrip(cdk2_pose_and_db):
    single, param_db = cdk2_pose_and_db
    pose_stack = PoseStackBuilder.from_poses(
        [single.clone(), single.clone(), single.clone()], single.device
    )
    ligand_block = _ligand_block(pose_stack)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)
    network = _network(pose_stack, param_db, docking_dofs)

    assert all(mapping.offset.shape == (3,) for mapping in docking_dofs.torsions)
    assert network.masked_dofs.numel() == 3 * (6 + len(docking_dofs.torsions))
    counts = torch.bincount(network.segment_ids, minlength=3)
    torch.testing.assert_close(counts, torch.full_like(counts, 8))
    rebuilt = network.pose_stack_from_dofs()
    error = float(torch.max(torch.abs(rebuilt.coords - pose_stack.coords)))
    assert error < 1.0e-9


def test_batched_distinct_torsions_are_pose_local(cdk2_pose_and_db):
    single, param_db = cdk2_pose_and_db
    identical_stack = PoseStackBuilder.from_poses(
        [single.clone(), single.clone(), single.clone()], single.device
    )
    ligand_block = _ligand_block(identical_stack)
    initial_docking_dofs = build_ligand_docking_dofs(
        identical_stack, ligand_block
    )
    initial_network = _network(
        identical_stack, param_db, initial_docking_dofs
    )

    # Give both logical torsions distinct values in every pose before testing
    # the pose-local response of the rebuilt batched mapping.
    per_pose_deltas = ((0.0, 0.0), (0.31, -0.26), (-0.47, 0.42))
    with torch.no_grad():
        for pose, pose_deltas in enumerate(per_pose_deltas):
            for mapping, delta in zip(
                initial_docking_dofs.torsions, pose_deltas
            ):
                column = _raw_dof_column(
                    initial_network,
                    identical_stack,
                    ligand_block,
                    mapping,
                    pose,
                )
                initial_network.masked_dofs[column] += delta
    distinct_stack = initial_network.pose_stack_from_dofs()

    docking_dofs = build_ligand_docking_dofs(distinct_stack, ligand_block)
    network = _network(distinct_stack, param_db, docking_dofs)
    base_dofs = network.masked_dofs.detach().double()
    base_torsions = _batched_torsion_values(
        network, distinct_stack, ligand_block
    )
    for torsion_index in range(base_torsions.shape[1]):
        pairwise = _angle_delta(
            base_torsions[:, torsion_index, None],
            base_torsions[None, :, torsion_index],
        )
        off_diagonal = pairwise[~torch.eye(3, dtype=torch.bool)]
        assert bool(torch.all(torch.abs(off_diagonal) > 0.1))

    intended_pose = 1
    intended_mapping = docking_dofs.torsions[1]
    column = _raw_dof_column(
        network,
        distinct_stack,
        ligand_block,
        intended_mapping,
        intended_pose,
    )
    epsilon = 1.0e-5
    perturbed = base_dofs.clone()
    perturbed[column] += epsilon
    changed_full_dofs = network.full_dofs.detach().reshape(-1).scatter(
        0, network._dof_flat_idx, perturbed
    ).view_as(network.full_dofs)
    changed_torsions = _batched_torsion_values(
        network, distinct_stack, ligand_block, changed_full_dofs
    )
    derivative = _angle_delta(changed_torsions, base_torsions) / epsilon
    expected = torch.zeros_like(derivative)
    expected[intended_pose, intended_mapping.torsion_index] = (
        intended_mapping.sign
    )
    torch.testing.assert_close(derivative, expected, rtol=1.0e-4, atol=1.0e-5)


def test_builder_does_not_mutate_shared_kinematics_or_ligand_root():
    # Load a genuinely fresh PBT that has not been passed through a
    # PoseStackKinematicsModule by another test.
    pose_stack, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(pose_stack)
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    pbt = pose_stack.packed_block_types

    plain_ff = FoldForest.reasonable_fold_forest(pose_stack)
    plain_mm = MoveMap.from_pose_stack(pose_stack)
    plain_mm.set_move_all_jump_dofs_for_root_jump(0, ligand_block, True)
    for torsion_index in range(block_type.n_torsions):
        plain_mm.set_move_named_torsion_for_blocks(
            0, ligand_block, torsion_index, True
        )

    # Snapshot what is already stored on the fresh objects. The builder is
    # allowed to populate missing ordinary kinematics caches lazily.
    fresh_pbt_scan_values = _scan_cache_values(pbt)
    fresh_block_scan_values = _scan_cache_values(block_type)
    jump_atom = block_type.default_jump_connection_atom
    jump_atom_index = block_type.default_jump_connection_atom_index
    assert not hasattr(pbt, "ligand_docking_dofs")
    assert not hasattr(block_type, "ligand_docking_dofs")

    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)

    assert (docking_dofs.fold_forest.edges == plain_ff.edges).all()
    assert (docking_dofs.fold_forest.n_edges == plain_ff.n_edges).all()
    assert block_type.default_jump_connection_atom == jump_atom == "N5"
    assert block_type.default_jump_connection_atom_index == jump_atom_index
    _assert_scan_cache_values_equal(pbt, fresh_pbt_scan_values)
    _assert_scan_cache_values_equal(block_type, fresh_block_scan_values)
    assert not hasattr(pbt, "ligand_docking_dofs")
    assert not hasattr(block_type, "ligand_docking_dofs")

    # Once the standard lazy caches exist, compare their values (not merely
    # object identity) across another docking build.
    pbt_scan_values = _scan_cache_values(pbt)
    block_scan_values = _scan_cache_values(block_type)
    assert pbt_scan_values is not None
    assert block_scan_values is not None
    plain_kin_before = PoseStackKinematicsModule(pose_stack, plain_ff)
    plain_mask_before = MinimizerMap(
        pose_stack, plain_kin_before.kmd, plain_mm
    ).dof_mask.clone()
    # The ordinary F22 path exposes chi1 through C7.phi_c but loses chi2 because
    # C4.phi_c is fixed by the jump stub: 6 jump + 1 torsion = 7. The docking
    # resolver recovers chi2 through C7.phi_p for a total of 8.
    plain_baseline = int(torch.sum(plain_mask_before))
    assert plain_baseline == 7

    docking_dofs_again = build_ligand_docking_dofs(pose_stack, ligand_block)

    assert block_type.default_jump_connection_atom == jump_atom == "N5"
    assert block_type.default_jump_connection_atom_index == jump_atom_index
    _assert_scan_cache_values_equal(pbt, pbt_scan_values)
    _assert_scan_cache_values_equal(block_type, block_scan_values)
    assert not hasattr(pbt, "ligand_docking_dofs")
    assert not hasattr(block_type, "ligand_docking_dofs")

    plain_kin_after = PoseStackKinematicsModule(pose_stack, plain_ff)
    plain_mask_after = MinimizerMap(
        pose_stack, plain_kin_after.kmd, plain_mm
    ).dof_mask
    torch.testing.assert_close(plain_mask_after, plain_mask_before)
    assert int(torch.sum(plain_mask_after)) == plain_baseline
    assert int(
        torch.sum(
            MinimizerMap(
                pose_stack,
                PoseStackKinematicsModule(
                    pose_stack, docking_dofs_again.fold_forest
                ).kmd,
                docking_dofs_again.move_map,
            ).dof_mask
        )
    ) == 8


def test_builder_batches_atom_lookup_and_empirical_signs(
    cdk2_pose_and_db, monkeypatch
):
    single, _ = cdk2_pose_and_db
    pose_stack = PoseStackBuilder.from_poses(
        [single.clone(), single.clone(), single.clone()], single.device
    )
    ligand_block = _ligand_block(pose_stack)
    calls = {"node_map": 0, "signs": 0}

    original_node_map = ligand_dofs_module._ligand_kinforest_nodes
    original_signs = ligand_dofs_module._empirical_torsion_signs

    def capture_node_map(*args, **kwargs):
        calls["node_map"] += 1
        result = original_node_map(*args, **kwargs)
        assert result.shape[0] == pose_stack.n_poses
        return result

    def capture_signs(
        kin_module,
        raw_dofs,
        ligand_nodes,
        torsion_atoms,
        raw_nodes,
        dof_types,
    ):
        calls["signs"] += 1
        assert raw_nodes.shape == (
            pose_stack.n_poses,
            len(torsion_atoms),
        )
        return original_signs(
            kin_module,
            raw_dofs,
            ligand_nodes,
            torsion_atoms,
            raw_nodes,
            dof_types,
        )

    monkeypatch.setattr(
        ligand_dofs_module, "_ligand_kinforest_nodes", capture_node_map
    )
    monkeypatch.setattr(
        ligand_dofs_module, "_empirical_torsion_signs", capture_signs
    )
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)

    # Each helper receives the complete pose/atom or pose/torsion tensor once.
    # The empirical helper internally performs the required one-at-a-time
    # torsion perturbations while retaining batched processing across poses.
    assert calls == {"node_map": 1, "signs": 1}
    assert len(docking_dofs.torsions) == 2
    node_map_source = inspect.getsource(
        ligand_dofs_module._ligand_kinforest_nodes
    )
    assert "nonzero" not in node_map_source
    assert ".item(" not in node_map_source
    assert "int(" not in node_map_source
