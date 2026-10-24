"""Phase-2 gates for the continuous batched docking loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tmol.dock.dock_loop as dock_loop
from tmol.database import ParameterDatabase
from tmol.dock import (
    ContinuousBackend,
    DockingPerturbation,
    ScoringBackend,
    build_ligand_docking_dofs,
    run_one_generation,
)
from tmol.dock.dock_loop import _inverse_dofs
from tmol.io import pose_stack_from_biotite
from tmol.kinematics import BondDOFTypes, JumpDOFTypes, NodeType
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.pose import PoseStackBuilder
from tmol.score import beta2016_score_function
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    PLI_DATA_DIR,
    _ligand_block,
    _load_cdk2_pose,
)
from tmol.tests.ligand.test_protein_ligand_ddg import _load_complex_cif


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def cdk2_cpu_pose_and_db():
    # This is the exact Phase-1 cdk2 loader required by the Phase-2 gate.
    return _load_cdk2_pose()


def _load_cdk2_pose_on_device(device: torch.device):
    if device.type == "cpu":
        return _load_cdk2_pose()

    # The Phase-1 helper is CPU/fp64-only. Repeat its exact construction on the
    # requested device for the GPU/fp32 batched lane.
    pose_stack, context = pose_stack_from_biotite(
        _load_complex_cif("cdk2"),
        device,
        prepare_ligands=True,
        ligand_params_files=[
            str(PLI_DATA_DIR / "cdk2.xtal-lig.mmff94.tmol")
        ],
        no_optH=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=torch.float32)
    return pose_stack, context.parameter_database


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("cpu", torch.float64, 2), id="cpu_fp64"),
        pytest.param(
            ("cuda", torch.float32, 4),
            id="cuda_fp32_npop4",
            marks=pytest.mark.skipif(
                not CUDA_AVAILABLE, reason="requires a CUDA device"
            ),
        ),
    ],
)
def completed_generation(request):
    device_name, dtype, n_pop = request.param
    device = torch.device(device_name)
    reference, param_db = _load_cdk2_pose_on_device(device)
    reference.coords = reference.coords.to(dtype=dtype)
    ligand_block = _ligand_block(reference)
    backend = ContinuousBackend(param_db, kin_dtype=dtype)
    result = run_one_generation(
        reference,
        ligand_block,
        backend,
        n_pop=n_pop,
        seed=20260821,
        perturb=DockingPerturbation(),
    )
    return SimpleNamespace(
        reference=reference,
        ligand_block=ligand_block,
        n_pop=n_pop,
        result=result,
    )


def _ligand_atom_range(pose_stack, ligand_block):
    offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[
        block_type_index
    ]
    return offset, block_type


def test_continuous_backend_score_is_weighted_beta16_total(
    cdk2_cpu_pose_and_db,
):
    reference, param_db = cdk2_cpu_pose_and_db
    pose_stack = PoseStackBuilder.from_poses(
        [reference, reference], reference.device
    )
    pose_stack.coords = reference.coords.repeat(2, 1, 1)
    assert pose_stack.device.type == "cpu"
    assert pose_stack.coords.dtype == torch.float64

    oracle_sfxn = beta2016_score_function(
        pose_stack.device, param_db=param_db
    )
    oracle_scorer = oracle_sfxn.render_whole_pose_scoring_module(pose_stack)
    with torch.no_grad():
        unweighted_term_rows = oracle_scorer(
            pose_stack.coords, sum_terms=False, apply_weights=False
        )
        weights = oracle_sfxn.weights_tensor().to(
            dtype=unweighted_term_rows.dtype
        )
        expected = torch.sum(
            weights[:, None] * unweighted_term_rows, dim=0
        )

    actual = ContinuousBackend(param_db).score(pose_stack)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-10)


def test_continuous_backend_runs_batched_and_selects(completed_generation):
    case = completed_generation
    result = case.result
    n_pop = case.n_pop

    assert result.minimized_pose_stack.n_poses == n_pop
    assert result.initial_energies.shape == (n_pop,)
    assert result.final_energies.shape == (n_pop,)
    assert result.selection.shape == (n_pop,)
    assert result.selection.dtype == torch.int64
    assert torch.isfinite(result.initial_energies).all()
    assert torch.isfinite(result.final_energies).all()
    assert result.minimized_pose_stack.coords.requires_grad is False

    selected = result.final_energies[result.selection]
    assert torch.all(selected[1:] >= selected[:-1])
    assert int(result.best_index) == int(torch.argmin(result.final_energies))
    assert int(result.selection[0]) == int(result.best_index)


def test_minimization_does_not_increase_per_pose(completed_generation):
    result = completed_generation.result
    decreases = result.initial_energies - result.final_energies
    # Separate numerical allowances for total-score reevaluation after the
    # minimizer: 1e-6 REU in fp64 and 5e-3 REU in fp32.
    eps = (
        1.0e-6
        if completed_generation.reference.coords.dtype == torch.float64
        else 5.0e-3
    )
    print(
        "cdk2 per-pose continuous minimization energy decreases:",
        decreases.detach().cpu().tolist(),
    )
    assert torch.all(result.final_energies <= result.initial_energies + eps)


def test_native_pose_stays_near_crystal(completed_generation):
    case = completed_generation
    minimized = case.result.minimized_pose_stack
    offset, block_type = _ligand_atom_range(
        case.reference, case.ligand_block
    )
    ligand_slice = slice(offset, offset + block_type.n_atoms)
    block_type_index = int(
        case.reference.block_type_ind64[0, case.ligand_block]
    )
    heavy = ~case.reference.packed_block_types.atom_is_hydrogen[
        block_type_index, : block_type.n_atoms
    ].to(torch.bool)
    delta = (
        minimized.coords[0, ligand_slice][heavy]
        - case.reference.coords[0, ligand_slice][heavy]
    )
    rmsd = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1)))
    native_energy = case.result.final_energies[0]
    print(
        f"cdk2 native-after-min ligand heavy-atom RMSD={float(rmsd):.6f} A "
        f"total_beta16={float(native_energy):.6f}"
    )

    # Starting from the crystal pose should remain in its local basin. The
    # initial Phase-2 bound is 0.75 A and can be tightened after Slurm data.
    assert float(rmsd) < 0.75
    assert torch.isfinite(native_energy)
    # A broad corruption guard: normal cdk2 total energies are orders of
    # magnitude smaller than one million REU.
    assert float(torch.abs(native_energy)) < 1.0e6


def test_minimization_moves_only_ligand(completed_generation):
    case = completed_generation
    minimized = case.result.minimized_pose_stack
    offset, block_type = _ligand_atom_range(
        case.reference, case.ligand_block
    )
    receptor = torch.ones(
        case.reference.max_n_pose_atoms,
        dtype=torch.bool,
        device=case.reference.device,
    )
    receptor[offset : offset + block_type.n_atoms] = False
    expected = case.reference.coords[:, receptor].expand(case.n_pop, -1, -1)
    actual = minimized.coords[:, receptor]

    # Fixed receptor coordinates undergo only inverse/forward kinematic
    # roundoff in run_kin_min (Phase 1 measured ~1e-9 fp64, ~1e-4 A fp32).
    atol = 2.0e-9 if actual.dtype == torch.float64 else 5.0e-4
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


class _CaptureIdentityBackend(ScoringBackend):
    def __init__(self):
        self.scored_coords = []

    def score(self, pose_stack):
        self.scored_coords.append(pose_stack.coords.detach().clone())
        return torch.zeros(
            pose_stack.n_poses,
            dtype=pose_stack.coords.dtype,
            device=pose_stack.device,
        )

    def minimize(self, pose_stack, docking_dofs):
        return pose_stack


def test_initial_dof_jitter_is_seeded_bounded_and_ligand_dof_only(
    cdk2_cpu_pose_and_db, monkeypatch
):
    reference, _ = cdk2_cpu_pose_and_db
    ligand_block = _ligand_block(reference)
    captured_kin_modules = []
    captured_forward_dofs = []

    class _CaptureForwardDOFs(PoseStackKinematicsModule):
        def __init__(self, pose_stack, fold_forest):
            super().__init__(pose_stack, fold_forest)
            captured_kin_modules.append(self)

        def forward(self, dofs):
            captured_forward_dofs.append(dofs.detach().clone())
            return super().forward(dofs)

    # Observe the exact raw-DOF tensor supplied to forward kinematics. Inverse
    # kinematics of the resulting coordinates is not a valid oracle here:
    # jump rotations and reversed ligand torsions have redundant/coupled raw
    # representations that inverse kinematics is free to redistribute.
    monkeypatch.setattr(
        dock_loop, "PoseStackKinematicsModule", _CaptureForwardDOFs
    )

    first_backend = _CaptureIdentityBackend()
    second_backend = _CaptureIdentityBackend()
    different_seed_backend = _CaptureIdentityBackend()
    perturb = DockingPerturbation()
    kwargs = dict(
        n_pop=4,
        seed=314159,
        perturb=perturb,
    )
    run_one_generation(reference, ligand_block, first_backend, **kwargs)
    run_one_generation(reference, ligand_block, second_backend, **kwargs)
    run_one_generation(
        reference,
        ligand_block,
        different_seed_backend,
        n_pop=4,
        seed=271828,
        perturb=perturb,
    )
    first = first_backend.scored_coords[0]
    second = second_backend.scored_coords[0]
    different_seed = different_seed_backend.scored_coords[0]

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first[0], reference.coords[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        different_seed[0], reference.coords[0], rtol=0.0, atol=0.0
    )
    assert not torch.equal(first[1:], different_seed[1:])

    offset, block_type = _ligand_atom_range(reference, ligand_block)
    receptor = torch.ones(
        reference.max_n_pose_atoms, dtype=torch.bool, device=reference.device
    )
    receptor[offset : offset + block_type.n_atoms] = False
    expected_receptor = reference.coords[:, receptor].expand(4, -1, -1)
    torch.testing.assert_close(
        first[:, receptor], expected_receptor, rtol=0.0, atol=0.0
    )
    ligand_slice = slice(offset, offset + block_type.n_atoms)
    assert torch.any(first[1:, ligand_slice] != first[0, ligand_slice])

    assert len(captured_kin_modules) == 3
    assert len(captured_forward_dofs) == 3
    first_forward_dofs, second_forward_dofs, different_seed_forward_dofs = (
        captured_forward_dofs
    )
    torch.testing.assert_close(
        first_forward_dofs, second_forward_dofs, rtol=0.0, atol=0.0
    )
    assert not torch.equal(first_forward_dofs, different_seed_forward_dofs)

    population = PoseStackBuilder.from_poses(
        [reference] * 4, reference.device
    )
    population.coords = reference.coords.repeat(4, 1, 1)
    docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    kin_module = captured_kin_modules[0]
    base_dofs = _inverse_dofs(population, kin_module)
    # This is the actual additive raw-DOF perturbation passed to forward
    # kinematics, before any redundant inverse-kinematic representation can
    # redistribute the coordinate change onto coupled columns.
    applied_delta = first_forward_dofs - base_dofs

    kmd = kin_module.kmd
    node_types = kmd.forest.doftype

    free = MinimizerMap(
        population, kmd, docking_dofs.move_map
    ).dof_mask
    pose_for_node = torch.div(
        kmd.forest.id,
        population.max_n_pose_atoms,
        rounding_mode="floor",
    )
    expected_changed = free & (pose_for_node[:, None] > 0)
    assert int(torch.sum(expected_changed)) == 3 * (
        6 + len(docking_dofs.torsions)
    )
    torch.testing.assert_close(
        first_forward_dofs[~expected_changed],
        base_dofs[~expected_changed],
        rtol=0.0,
        atol=0.0,
    )
    assert bool(torch.all(torch.abs(applied_delta[expected_changed]) > 0.0))

    changed_nodes, changed_columns = torch.nonzero(
        expected_changed, as_tuple=True
    )
    changed_node_types = node_types[changed_nodes]
    translation = (changed_node_types == int(NodeType.jump)) & (
        changed_columns <= int(JumpDOFTypes.RBz)
    )
    rotation = (
        (changed_node_types == int(NodeType.jump))
        & (changed_columns >= int(JumpDOFTypes.RBdel_alpha))
        & (changed_columns <= int(JumpDOFTypes.RBdel_gamma))
    )
    torsion = (changed_node_types == int(NodeType.bond)) & (
        (changed_columns == int(BondDOFTypes.phi_p))
        | (changed_columns == int(BondDOFTypes.phi_c))
    )
    assert bool(torch.all(translation | rotation | torsion))
    assert int(torch.sum(translation)) == 3 * 3
    assert int(torch.sum(rotation)) == 3 * 3
    assert int(torch.sum(torsion)) == 3 * len(docking_dofs.torsions)

    changed_delta = torch.abs(applied_delta[changed_nodes, changed_columns])
    numerical_margin = 1.0e-12
    assert bool(
        torch.all(
            changed_delta[translation]
            <= perturb.translation_angstrom + numerical_margin
        )
    )
    assert bool(
        torch.all(
            changed_delta[rotation]
            <= perturb.rotation_radians + numerical_margin
        )
    )
    assert bool(
        torch.all(
            changed_delta[torsion]
            <= perturb.torsion_radians + numerical_margin
        )
    )


