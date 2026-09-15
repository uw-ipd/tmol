"""M4 gates for vectorized, device-resident GALigandDock operators."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import attr
import pytest
import torch

import tmol.dock.ga as ga_module
import tmol.dock.scoring_backend as scoring_backend_module
from tmol.dock import (
    ContinuousBackend,
    GAPool,
    GASchedule,
    LigandGeneLayout,
    MinimizationResult,
    ScoringBackend,
    build_ligand_docking_dofs,
    crossover,
    ligand_heavy_atom_rms,
    mutate,
    run_ga_docking,
    update_pool,
)
from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.tests.dock.test_dock_loop import _load_cdk2_pose_on_device
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def cpu_ga_case():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    population = _repeat_pose_stack(reference, 6)
    docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    return SimpleNamespace(
        reference=reference,
        param_db=param_db,
        ligand_block=ligand_block,
        population=population,
        docking_dofs=docking_dofs,
        layout=layout,
    )


def _receptor_mask(case):
    block_type_index = int(
        case.population.block_type_ind64[0, case.ligand_block]
    )
    block_type = case.population.packed_block_types.active_block_types[
        block_type_index
    ]
    offset = int(
        case.population.block_coord_offset64[0, case.ligand_block]
    )
    receptor = torch.ones(
        case.population.max_n_pose_atoms,
        dtype=torch.bool,
        device=case.population.device,
    )
    receptor[offset : offset + block_type.n_atoms] = False
    return receptor


def _oracle_column_zyx_rotation(euler):
    alpha, beta, gamma = torch.unbind(euler, dim=-1)
    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    rx = torch.stack(
        (
            ones,
            zeros,
            zeros,
            zeros,
            torch.cos(alpha),
            -torch.sin(alpha),
            zeros,
            torch.sin(alpha),
            torch.cos(alpha),
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))
    ry = torch.stack(
        (
            torch.cos(beta),
            zeros,
            torch.sin(beta),
            zeros,
            ones,
            zeros,
            -torch.sin(beta),
            zeros,
            torch.cos(beta),
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))
    rz = torch.stack(
        (
            torch.cos(gamma),
            -torch.sin(gamma),
            zeros,
            torch.sin(gamma),
            torch.cos(gamma),
            zeros,
            zeros,
            zeros,
            ones,
        ),
        dim=-1,
    ).reshape(euler.shape[:-1] + (3, 3))
    return rz @ ry @ rx


def _oracle_quaternion_rotation(quaternion):
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def test_rotation_mutation_matches_rosetta_quaternion_oracle():
    dtype = torch.float64
    parents = torch.tensor(
        [[0.0, 0.0, 0.0], [0.23, -0.31, 0.47]], dtype=dtype
    )
    axes = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=dtype
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=1, keepdim=True)
    angles = torch.tensor([0.19, -0.37], dtype=dtype)

    mutated_euler = ga_module._apply_rosetta_rotation_mutation(
        parents, axes, angles
    )
    actual = ga_module._zyx_rotation_matrix(mutated_euler).transpose(-1, -2)

    # LigandConformer.cc:775-779 uses q=(cos(a), sin(a)*axis), then
    # Qnew=Qperturb*Qparent. Thus its physical perturbation angle is 2*a.
    perturbation_quaternion = torch.cat(
        (torch.cos(angles)[:, None], torch.sin(angles)[:, None] * axes),
        dim=1,
    )
    expected = _oracle_quaternion_rotation(
        perturbation_quaternion
    ) @ _oracle_column_zyx_rotation(parents)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-9)


def test_crossover_forces_at_least_one_partner_chi():
    n_genes = 4096
    parent1 = torch.zeros((n_genes, 10), dtype=torch.float64)
    parent2 = torch.zeros_like(parent1)
    parent2[:, 6:] = torch.tensor([0.4, 0.8, 1.2, 1.6])

    offspring = crossover(
        parent1,
        parent2,
        generator=torch.Generator().manual_seed(20260821),
    )
    parent1_c = ga_module._canonicalize_genes(parent1)
    parent2_c = ga_module._canonicalize_genes(parent2)
    partner2_chi = offspring[:, 6:] == parent2_c[:, 6:]

    assert torch.all(torch.any(partner2_chi, dim=1))
    assert not torch.any(
        torch.all(offspring[:, 6:] == parent1_c[:, 6:], dim=1)
    )


def test_initial_freed_dof_gene_roundtrips_by_forward_kinematics(cpu_ga_case):
    case = cpu_ga_case
    genes = case.layout.genes_from_pose_stack(case.population)
    reconstructed = case.layout.pose_stack_from_genes(genes)

    assert genes.shape == (case.population.n_poses, 8)
    torch.testing.assert_close(
        reconstructed.coords, case.population.coords, rtol=0.0, atol=1.0e-9
    )


def test_mutate_and_crossover_are_finite_and_ligand_only(cpu_ga_case):
    case = cpu_ga_case
    genes = case.layout.genes_from_pose_stack(case.population)
    parent2 = genes.clone()
    parent2[:, :3] += torch.tensor(
        [1.5, -0.75, 0.25], dtype=genes.dtype, device=genes.device
    )
    parent2[:, 3:6] += torch.tensor(
        [0.2, -0.15, 0.35], dtype=genes.dtype, device=genes.device
    )
    parent2[:, 6:] += 0.6

    generator = torch.Generator(device=genes.device).manual_seed(20260821)
    mutated = mutate(genes, generator=generator)
    crossed = crossover(genes, parent2, generator=generator)
    mutated_pose = case.layout.pose_stack_from_genes(mutated)
    crossed_pose = case.layout.pose_stack_from_genes(crossed)

    assert mutated.shape == genes.shape == crossed.shape
    assert mutated.device == genes.device == crossed.device
    assert torch.isfinite(mutated).all()
    assert torch.isfinite(crossed).all()
    assert torch.isfinite(mutated_pose.coords).all()
    assert torch.isfinite(crossed_pose.coords).all()
    assert torch.all(torch.any(mutated != genes, dim=1))

    # Crossover returns a canonical gene with wrapped torsions, so an inherited
    # chi matches the canonicalized parent rather than an out-of-range value.
    genes_c = ga_module._canonicalize_genes(genes)
    parent2_c = ga_module._canonicalize_genes(parent2)
    rb_from_parent = torch.all(
        crossed[:, :6] == genes_c[:, :6], dim=1
    ) | torch.all(
        crossed[:, :6] == parent2_c[:, :6], dim=1
    )
    assert torch.all(rb_from_parent)
    if genes.shape[1] > 6:
        assert torch.all(
            (crossed[:, 6:] == genes_c[:, 6:])
            | (crossed[:, 6:] == parent2_c[:, 6:])
        )

    receptor = _receptor_mask(case)
    expected = case.population.coords[:, receptor]
    torch.testing.assert_close(
        mutated_pose.coords[:, receptor], expected, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        crossed_pose.coords[:, receptor], expected, rtol=0.0, atol=0.0
    )


def test_minimize_returns_gene_from_optimizer_masked_dofs(
    cpu_ga_case, monkeypatch
):
    case = cpu_ga_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[:, :6] += torch.tensor(
        [0.3, -0.2, 0.1, 0.25, -0.15, 0.05], dtype=genes.dtype
    )
    genes[:, 6:] += 0.4
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    proposed = case.layout.pose_stack_from_genes(genes)
    captured = {}

    def capture_noop_min(kf_network, **kwargs):
        captured["network"] = kf_network
        return kf_network.pose_stack_from_dofs()

    monkeypatch.setattr(
        scoring_backend_module, "run_min", capture_noop_min
    )
    result = ContinuousBackend(
        case.param_db, kin_dtype=torch.float64
    ).minimize(
        proposed,
        case.docking_dofs,
        maxiter=1,
        initial_raw_dofs=raw_dofs,
    )
    network = captured["network"]
    optimizer_raw_dofs = network.full_dofs.detach().clone()
    optimizer_raw_dofs[network.dof_mask] = result.masked_dofs
    expected_genes = case.layout.genes_from_raw_dofs(optimizer_raw_dofs)

    torch.testing.assert_close(
        result.masked_dofs,
        raw_dofs[network.dof_mask],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.genes, expected_genes, rtol=0.0, atol=1.0e-12
    )
    torch.testing.assert_close(result.genes, genes, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(
        result.pose_stack.coords, proposed.coords, rtol=0.0, atol=1.0e-9
    )
    rebuilt = case.layout.pose_stack_from_genes(result.genes)
    torch.testing.assert_close(
        rebuilt.coords, result.pose_stack.coords, rtol=0.0, atol=1.0e-9
    )


def test_distance_matches_independent_heavy_atom_rms(cpu_ga_case):
    case = cpu_ga_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[:, 0] += torch.arange(
        genes.shape[0], dtype=genes.dtype, device=genes.device
    )
    poses = case.layout.pose_stack_from_genes(genes)
    heavy = case.layout.ligand_heavy_coords(poses)

    actual = ligand_heavy_atom_rms(heavy[:, None], heavy[None, :])
    delta = heavy[:, None, :, :] - heavy[None, :, :, :]
    expected = torch.sqrt(torch.sum(delta**2, dim=(2, 3)) / heavy.shape[1])
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_update_pool_enforces_diversity_and_documents_fill(cpu_ga_case):
    case = cpu_ga_case
    layout = case.layout
    base = layout.genes_from_pose_stack(case.population)
    parent_genes = base[:3].clone()
    offspring_genes = base[:3].clone()
    parent_genes[:, 0] += torch.tensor(
        [0.0, 6.0, 12.0], dtype=base.dtype, device=base.device
    )
    offspring_genes[:, 0] += torch.tensor(
        [3.0, 9.0, 15.0], dtype=base.dtype, device=base.device
    )

    # update_pool requires matching npool topology; use three-pose layouts.
    population = _repeat_pose_stack(case.reference, 3)
    docking_dofs = build_ligand_docking_dofs(population, case.ligand_block)
    layout3 = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    parents = GAPool(
        genes=parent_genes,
        pose_stack=layout3.pose_stack_from_genes(parent_genes),
        energies=torch.tensor([0.0, 2.0, 4.0], dtype=base.dtype),
        n_diverse=3,
    )
    offspring = GAPool(
        genes=offspring_genes,
        pose_stack=layout3.pose_stack_from_genes(offspring_genes),
        energies=torch.tensor([1.0, 3.0, 5.0], dtype=base.dtype),
        n_diverse=3,
    )
    selected = update_pool(
        parents, offspring, layout3, npool=3, rmsdthreshold=2.0
    )
    heavy = layout3.ligand_heavy_coords(selected.pose_stack)
    distances = ligand_heavy_atom_rms(heavy[:, None], heavy[None, :])
    off_diagonal = distances[~torch.eye(3, dtype=torch.bool)]
    assert selected.genes.shape[0] == 3
    assert selected.n_diverse == 3
    assert torch.all(off_diagonal >= 2.0)

    duplicate_parent = replace(
        parents,
        genes=base[:3],
        pose_stack=layout3.pose_stack_from_genes(base[:3]),
    )
    duplicate_offspring = replace(
        offspring,
        genes=base[:3],
        pose_stack=layout3.pose_stack_from_genes(base[:3]),
    )
    filled = update_pool(
        duplicate_parent,
        duplicate_offspring,
        layout3,
        npool=3,
        rmsdthreshold=2.0,
    )
    assert filled.genes.shape[0] == 3
    assert filled.n_diverse == 1


@pytest.mark.parametrize(
    "energy_values,x_values,expect_diverse",
    [
        ([4.0, 0.0, 3.0, 2.0, 1.0, 5.0], [0.0, 0.5, 3.0, 6.0, 6.2, 9.0], True),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], False),
    ],
)
def test_diverse_low_energy_selection_matches_independent_greedy_reference(
    energy_values, x_values, expect_diverse
):
    energies = torch.tensor(
        energy_values,
        dtype=torch.float64,
    )
    x_positions = torch.tensor(
        x_values,
        dtype=torch.float64,
    )
    coords = torch.zeros((6, 2, 3), dtype=torch.float64)
    coords[:, :, 0] = x_positions[:, None]
    npool = 4
    threshold = 2.0

    def independent_reference():
        accepted = []
        rejected = []
        for candidate in torch.argsort(energies).tolist():
            candidate_coords = coords[candidate]
            sufficiently_distant = all(
                float(
                    torch.sqrt(
                        torch.mean(
                            torch.sum(
                                (candidate_coords - coords[previous]) ** 2,
                                dim=1,
                            )
                        )
                    )
                )
                >= threshold
                for previous in accepted
            )
            if sufficiently_distant and len(accepted) < npool:
                accepted.append(candidate)
            else:
                rejected.append(candidate)
        return accepted + rejected[: npool - len(accepted)], len(accepted)

    expected = independent_reference()
    selected, n_diverse = ga_module._diverse_low_energy_indices(
        energies, coords, npool=npool, rmsdthreshold=threshold
    )

    torch.testing.assert_close(
        selected,
        torch.tensor(expected[0], dtype=torch.int64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        n_diverse,
        torch.tensor(expected[1], dtype=torch.int64),
        rtol=0.0,
        atol=0.0,
    )
    selected_coords = coords[selected]
    distances = ligand_heavy_atom_rms(
        selected_coords[:, None], selected_coords[None, :]
    )
    off_diagonal = ~torch.eye(npool, dtype=torch.bool)
    if expect_diverse:
        assert torch.all(distances[off_diagonal] >= threshold)


def test_swarm_subsample_is_exactly_reproducible(cpu_ga_case):
    case = cpu_ga_case
    swarm_size, npool = 6, 3
    swarm_population = _repeat_pose_stack(case.reference, swarm_size)
    swarm_dofs = build_ligand_docking_dofs(
        swarm_population, case.ligand_block
    )
    swarm_layout = LigandGeneLayout.from_pose_stack(
        swarm_population, swarm_dofs
    )
    pool_population = _repeat_pose_stack(case.reference, npool)
    pool_dofs = build_ligand_docking_dofs(pool_population, case.ligand_block)
    pool_layout = LigandGeneLayout.from_pose_stack(pool_population, pool_dofs)

    genes = swarm_layout.genes_from_pose_stack(swarm_population)
    local = torch.arange(swarm_size, dtype=genes.dtype)
    genes[:, 0] += local
    pose_stack = swarm_layout.pose_stack_from_genes(genes)
    energies = torch.tensor(
        [5.0, 0.0, 4.0, 1.0, 3.0, 2.0], dtype=genes.dtype
    )
    swarm = GAPool(
        genes=genes,
        pose_stack=pose_stack,
        energies=energies,
        n_diverse=swarm_size,
    )

    def subsample():
        return ga_module._subsample_swarm_pool(
            swarm,
            swarm_layout,
            pool_layout,
            swarm_size=swarm_size,
            npool=npool,
            rmsdthreshold=1.5,
        )

    selected = subsample()
    repeated = subsample()
    assert selected.genes.shape == (npool, genes.shape[1])
    torch.testing.assert_close(
        selected.genes, repeated.genes, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        selected.selected_indices,
        repeated.selected_indices,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        selected.selected_indices, torch.tensor([1, 3, 5]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(selected.genes, genes[[1, 3, 5]], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        selected.energies, energies[[1, 3, 5]], rtol=0.0, atol=0.0
    )
    assert selected.n_diverse == npool


class _TranslationImprovingBackend(ScoringBackend):
    def __init__(self, reference, ligand_block):
        block_type_index = int(reference.block_type_ind64[0, ligand_block])
        block_type = reference.packed_block_types.active_block_types[
            block_type_index
        ]
        self.offset = int(reference.block_coord_offset64[0, ligand_block])
        self.n_atoms = block_type.n_atoms
        ligand = reference.coords[
            0, self.offset : self.offset + self.n_atoms
        ]
        self.target_com = torch.mean(ligand, dim=0) + torch.tensor(
            [10.0, 0.0, 0.0], dtype=ligand.dtype, device=ligand.device
        )
        self.maxiters = []
        self.population_sizes = []
        self.minimized_genes = []

    def score(self, pose_stack):
        ligand = pose_stack.coords[
            :, self.offset : self.offset + self.n_atoms
        ]
        delta = torch.mean(ligand, dim=1) - self.target_com
        return torch.sum(delta * delta, dim=1)

    def minimize(
        self,
        pose_stack,
        docking_dofs,
        *,
        maxiter=None,
        initial_raw_dofs=None,
    ):
        self.maxiters.append(maxiter)
        self.population_sizes.append(pose_stack.n_poses)
        coords = pose_stack.coords.clone()
        ligand = coords[:, self.offset : self.offset + self.n_atoms]
        shift = 0.5 * (self.target_com - torch.mean(ligand, dim=1))
        coords[:, self.offset : self.offset + self.n_atoms] = (
            ligand + shift[:, None, :]
        )
        layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
        genes = layout.genes_from_raw_dofs(initial_raw_dofs).clone()
        genes[:, :3] += shift
        self.minimized_genes.append(genes.detach().clone())
        return MinimizationResult(
            pose_stack=attr.evolve(pose_stack, coords=coords),
            genes=genes,
            masked_dofs=None,
        )


def test_generation_loop_best_energy_is_monotone_and_strictly_improves():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    backend = _TranslationImprovingBackend(reference, ligand_block)
    schedule = GASchedule(
        generations=2,
        npool=4,
        rmsdthreshold=0.0,
        pmut=0.0,
        maxiter=3,
    )
    result = run_ga_docking(
        reference,
        ligand_block,
        backend,
        schedule=schedule,
        seed=7,
    )

    history = result.best_energy_history
    assert result.pool.genes.shape == (schedule.npool, 8)
    assert torch.all(history[1:] <= history[:-1])
    assert torch.any(history[1:] < history[0])
    assert backend.maxiters == [schedule.maxiter] * (schedule.generations + 1)


def test_swarm_initialization_runs_wide_gen0_then_small_ga_deterministically(
    monkeypatch,
):
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    schedule = GASchedule(
        generations=1,
        npool=3,
        rmsdthreshold=1.0,
        pmut=0.2,
        maxiter=2,
    )

    initial_ga_pools = []
    original_next_generation = ga_module.next_generation

    def capture_initial_pool(genes, *args, **kwargs):
        initial_ga_pools.append(genes.detach().clone())
        return original_next_generation(genes, *args, **kwargs)

    monkeypatch.setattr(ga_module, "next_generation", capture_initial_pool)

    def run_once(input_pose):
        backend = _TranslationImprovingBackend(reference, ligand_block)
        result = run_ga_docking(
            input_pose,
            ligand_block,
            backend,
            schedule=schedule,
            seed=20260827,
            initialization="swarm",
            swarm_size=8,
        )
        return result, backend

    result, backend = run_once(reference)
    repeated, repeated_backend = run_once(reference)
    assert result.pool.genes.shape == (schedule.npool, 8)
    assert backend.maxiters == [schedule.maxiter] * (schedule.generations + 1)
    assert backend.population_sizes == [
        8,
        schedule.npool,
    ]
    assert repeated_backend.maxiters == backend.maxiters
    torch.testing.assert_close(
        backend.minimized_genes[0],
        repeated_backend.minimized_genes[0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        initial_ga_pools[0], initial_ga_pools[1], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.pool.genes, repeated.pool.genes, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.pool.energies, repeated.pool.energies, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.pool.pose_stack.coords,
        repeated.pool.pose_stack.coords,
        rtol=0.0,
        atol=0.0,
    )


def test_inactive_swarm_size_does_not_change_random_initialization():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    schedule = GASchedule(generations=1, npool=3, rmsdthreshold=0.0, maxiter=1)

    def run_once(**kwargs):
        return run_ga_docking(
            reference,
            ligand_block,
            _TranslationImprovingBackend(reference, ligand_block),
            schedule=schedule,
            seed=17,
            initialization="random",
            **kwargs,
        )

    baseline = run_once()
    inactive = run_once(swarm_size=0)
    torch.testing.assert_close(
        baseline.pool.genes, inactive.pool.genes, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        baseline.pool.energies, inactive.pool.energies, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        baseline.pool.pose_stack.coords,
        inactive.pool.pose_stack.coords,
        rtol=0.0,
        atol=0.0,
    )


def test_swarm_size_must_cover_the_small_ga_pool():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    with pytest.raises(
        ValueError, match="swarm_size must be an integer at least schedule.npool"
    ):
        run_ga_docking(
            reference,
            ligand_block,
            _TranslationImprovingBackend(reference, ligand_block),
            schedule=GASchedule(generations=1, npool=4, maxiter=1),
            initialization="swarm",
            swarm_size=3,
        )


def test_hot_operators_have_no_python_pose_loop_or_scalar_sync():
    loop_free = (
        ga_module._canonical_torsions,
        ga_module._canonicalize_genes,
        ga_module._zyx_rotation_matrix,
        ga_module._rotation_matrix_to_zyx,
        ga_module._quaternion_rotation_matrix,
        ga_module._apply_rosetta_rotation_mutation,
        ga_module.mutate,
        ga_module.crossover,
        ga_module.ligand_heavy_atom_rms,
        ga_module.next_generation,
        ga_module.optimize_generation,
        ga_module.LigandGeneLayout.genes_from_raw_dofs,
        ga_module.LigandGeneLayout.genes_from_pose_stack,
        ga_module.LigandGeneLayout.raw_dofs_from_genes,
        ga_module.LigandGeneLayout.pose_stack_from_genes,
        ga_module.LigandGeneLayout.ligand_heavy_coords,
    )
    for function in loop_free:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)
        )
        assert ".item(" not in source
        assert ".tolist(" not in source

    optimize_source = textwrap.dedent(
        inspect.getsource(ga_module.optimize_generation)
    )
    assert "genes_from_pose_stack" not in optimize_source
    assert "_inverse_dofs" not in optimize_source

    # Both bounded host loops perform the same greedy niching rule. Keeping
    # update_pool unchanged preserves inactive initialization paths exactly;
    # the swarm loop bounds RMS storage by the small output pool.
    selection_tree = ast.parse(
        textwrap.dedent(inspect.getsource(ga_module._diverse_low_energy_indices))
    )
    assert sum(isinstance(node, ast.For) for node in ast.walk(selection_tree)) == 1
    update_tree = ast.parse(
        textwrap.dedent(inspect.getsource(ga_module.update_pool))
    )
    assert sum(isinstance(node, ast.For) for node in ast.walk(update_tree)) == 1
    run_source = textwrap.dedent(inspect.getsource(ga_module.run_ga_docking))
    run_tree = ast.parse(run_source)
    assert sum(isinstance(node, ast.For) for node in ast.walk(run_tree)) == 1
    assert ".item(" not in run_source
    assert ".tolist(" not in run_source


def test_cpu_fp64_continuous_ga_smoke():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    result = run_ga_docking(
        reference,
        ligand_block,
        ContinuousBackend(param_db, kin_dtype=torch.float64),
        schedule=GASchedule(
            generations=1,
            npool=2,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=2,
        ),
        seed=20260821,
    )
    assert result.pool.genes.shape == (2, 8)
    assert result.pool.genes.dtype == torch.float64
    assert torch.isfinite(result.pool.genes).all()
    assert torch.isfinite(result.pool.energies).all()
    assert torch.all(
        result.best_energy_history[1:] <= result.best_energy_history[:-1]
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
def test_gpu_fp32_cdk2_default_vsx_schedule():
    device = torch.device("cuda")
    reference, param_db = _load_cdk2_pose_on_device(device)
    ligand_block = _ligand_block(reference)
    schedule = GASchedule()
    result = run_ga_docking(
        reference,
        ligand_block,
        ContinuousBackend(param_db, kin_dtype=torch.float32),
        schedule=schedule,
        seed=20260821,
    )

    history = result.best_energy_history
    assert result.pool.genes.shape == (schedule.npool, 8)
    assert result.pool.genes.device.type == "cuda"
    assert result.pool.genes.dtype == torch.float32
    assert history.shape == (schedule.generations + 1,)
    assert torch.all(history[1:] <= history[:-1])
    assert torch.any(history[1:] < history[0])
    assert torch.isfinite(result.pool.pose_stack.coords).all()

    if result.pool.n_diverse == schedule.npool:
        layout = LigandGeneLayout.from_pose_stack(
            result.pool.pose_stack,
            build_ligand_docking_dofs(result.pool.pose_stack, ligand_block),
        )
        heavy = layout.ligand_heavy_coords(result.pool.pose_stack)
        distances = ligand_heavy_atom_rms(heavy[:, None], heavy[None, :])
        off_diagonal = distances[
            ~torch.eye(schedule.npool, dtype=torch.bool, device=device)
        ]
        assert torch.all(off_diagonal >= schedule.rmsdthreshold)
    else:
        # Rosetta GAOptimizer.cc:306-313 fills rejected candidates rather than
        # returning fewer than npool when the diversity cutoff cannot be met.
        assert result.pool.n_diverse < schedule.npool
    assert result.pool.genes.shape[0] == schedule.npool


@pytest.mark.parametrize(
    "initialization", ["jitter", "random", "swarm", "conformer", "aligner"]
)
def test_initialization_requires_one_reference_pose(cpu_ga_case, initialization):
    case = cpu_ga_case
    with pytest.raises(ValueError, match="must contain one reference pose"):
        run_ga_docking(
            _repeat_pose_stack(case.reference, 2),
            case.ligand_block,
            _TranslationImprovingBackend(case.reference, case.ligand_block),
            initialization=initialization,
        )
