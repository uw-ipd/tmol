"""Correctness gates for grid-guided rigid-body ligand alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tmol.dock.ga as ga_module
from tmol.dock import (
    ContinuousBackend,
    GridBackend,
    LigandAligner,
    LigandGeneLayout,
    RandomPlacement,
    build_ligand_docking_dofs,
    run_ga_docking,
)
from tmol.dock.ligand_aligner import (
    _axis_angle_rotation_matrix,
    _select_diverse_indices,
)
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)
from tmol.pose import repeat_pose_stack
from tmol.score.gridscore._movable import (
    movable_pose_stack,
)


@pytest.fixture(scope="module")
def cpu_aligner_case():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    backend = GridBackend.from_pose_stack(
        reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float64,
    ).for_ligand_block(0)
    ligand = movable_pose_stack(reference, ligand_block)
    return SimpleNamespace(
        reference=reference,
        param_db=param_db,
        ligand_block=ligand_block,
        backend=backend,
        ligand=ligand,
    )


def _layout(case, npool=8):
    population = repeat_pose_stack(case.ligand, npool)
    docking_dofs = build_ligand_docking_dofs(population, 0)
    return LigandGeneLayout.from_pose_stack(population, docking_dofs)


def _initialize(layout, backend, aligner, seed):
    generator = torch.Generator(device=layout.template.device).manual_seed(seed)
    random_genes = ga_module._random_initial_genes(
        layout,
        pocket_bounds=None,
        placement=RandomPlacement(),
        generator=generator,
    )
    aligned = aligner.seed_genes(
        layout,
        backend,
        random_genes,
        pocket_bounds=None,
        generator=generator,
    )
    return random_genes, aligned


@pytest.mark.parametrize(
    "threshold,expected", [(0.0, [0, 1]), (1.0, [0, 2]), (100.0, [0, 1])]
)
def test_absolute_rmsd_diversity_keeps_distinct_placements(threshold, expected):
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]], [[5.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    energies = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    selected = _select_diverse_indices(coordinates, energies, 2, threshold)
    torch.testing.assert_close(
        selected, torch.tensor(expected, dtype=torch.int64), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_candidates": 0}, "n_candidates"),
        ({"seed_fraction": -0.1}, "seed_fraction"),
        ({"seed_fraction": 1.1}, "seed_fraction"),
        ({"cluster_rmsd": -1.0}, "cluster_rmsd"),
        ({"refinement_steps": -1}, "refinement_steps"),
        ({"refinement_learning_rate": 0.0}, "refinement_learning_rate"),
    ],
)
def test_ligand_aligner_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LigandAligner(**kwargs)


def test_axis_angle_rotation_gradient_and_rigid_refinement():
    rotation_vector = torch.tensor(
        [[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        _axis_angle_rotation_matrix,
        (rotation_vector,),
        eps=1.0e-6,
        atol=1.0e-6,
        rtol=1.0e-5,
    )

    class QuadraticBackend:
        @staticmethod
        def _score_ligand_coordinates(coordinates):
            return torch.sum(coordinates * coordinates, dim=(1, 2))

    coordinates = torch.tensor(
        [[[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    atom_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    aligner = LigandAligner(
        n_candidates=1,
        seed_fraction=1.0,
        refinement_steps=10,
        refinement_top_k=1,
        refinement_learning_rate=0.1,
    )
    refined = aligner._refine_candidates(
        QuadraticBackend(), coordinates, atom_indices
    )
    original_energy = QuadraticBackend._score_ligand_coordinates(coordinates)[0]
    refined_energy = QuadraticBackend._score_ligand_coordinates(refined)[0]
    assert refined_energy < original_energy
    torch.testing.assert_close(
        torch.cdist(refined, refined),
        torch.cdist(coordinates, coordinates),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_candidate_generation_and_scoring_are_streamed():
    class TrackingBackend:
        max_batch = 0

        @classmethod
        def _score_ligand_coordinates(cls, coordinates):
            cls.max_batch = max(cls.max_batch, coordinates.shape[0])
            return torch.sum(coordinates * coordinates, dim=(1, 2))

    aligner = LigandAligner(
        n_candidates=5,
        score_batch_size=3,
        diversity_pool_size=2,
        refinement_top_k=1,
    )
    template = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    ligand_indices = torch.tensor([0, 1], dtype=torch.int64)
    heavy_indices = torch.tensor([0, 1], dtype=torch.int64)
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64
    )
    unit_positions = torch.zeros((5, 3), dtype=torch.float64)
    unit_positions[:, 0] = torch.arange(5, dtype=torch.float64) / 4
    rotations = torch.eye(3, dtype=torch.float64)[None].expand(5, -1, -1)
    centroids, selected_rotations, energies, candidate_indices = (
        aligner._score_and_preselect(
            TrackingBackend(),
            template,
            ligand_indices,
            heavy_indices,
            bounds,
            unit_positions,
            rotations,
            preselect_count=2,
        )
    )
    assert centroids.shape == (2, 3)
    assert selected_rotations.shape == (2, 3, 3)
    assert energies.shape == (2,)
    assert candidate_indices.shape == (2,)
    assert TrackingBackend.max_batch <= aligner.score_batch_size
    torch.testing.assert_close(candidate_indices, torch.tensor([0, 1]), rtol=0, atol=0)
    torch.testing.assert_close(
        energies, torch.tensor([0.5, 2.5], dtype=torch.float64), rtol=0, atol=0
    )


def _assert_seeded_coordinates_match_alignment(
    layout, backend, aligner, seed, n_seeded
):
    _, genes = _initialize(layout, backend, aligner, seed)
    decoded = layout.pose_stack_from_genes(genes)
    expected_generator = torch.Generator().manual_seed(seed)
    ga_module._random_initial_genes(
        layout,
        pocket_bounds=None,
        placement=RandomPlacement(),
        generator=expected_generator,
    )
    expected_alignment = aligner.align(
        layout,
        backend,
        n_placements=n_seeded,
        pocket_bounds=None,
        generator=expected_generator,
    )
    coordinate_error = torch.max(
        torch.abs(
            decoded.coords[1 : n_seeded + 1] - expected_alignment.coordinates
        )
    )
    return genes, expected_alignment, coordinate_error


def test_aligner_seed_genes_are_deterministic_rigid_and_diverse(cpu_aligner_case):
    case = cpu_aligner_case
    layout = _layout(case)
    aligner = LigandAligner(
        n_candidates=32,
        seed_fraction=0.5,
        cluster_rmsd=0.5,
        diversity_pool_size=16,
        score_batch_size=16,
        refinement_steps=0,
    )

    random_genes, genes = _initialize(layout, case.backend, aligner, 20260826)
    repeat_random, repeat = _initialize(layout, case.backend, aligner, 20260826)
    torch.testing.assert_close(random_genes, repeat_random, rtol=0.0, atol=0.0)
    torch.testing.assert_close(genes, repeat, rtol=0.0, atol=0.0)

    reference = layout.genes_from_pose_stack(layout.template)
    torch.testing.assert_close(genes[0], reference[0], rtol=0.0, atol=0.0)
    n_seeded = 4
    torch.testing.assert_close(
        genes[1 : n_seeded + 1, 6:],
        reference[1 : n_seeded + 1, 6:],
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        genes[n_seeded + 1 :],
        random_genes[n_seeded + 1 :],
        rtol=0.0,
        atol=0.0,
    )

    _, expected_alignment, coordinate_error = (
        _assert_seeded_coordinates_match_alignment(
            layout, case.backend, aligner, 20260826, n_seeded
        )
    )
    decoded = layout.pose_stack_from_genes(genes)
    assert torch.all(
        expected_alignment.energies[1:] >= expected_alignment.energies[:-1]
    )
    print(f"aligner gene decode max coordinate error={float(coordinate_error):.3e} A")
    assert float(coordinate_error) <= 2.0e-12

    placed = layout.ligand_heavy_coords(decoded)
    template = layout.ligand_heavy_coords(layout.template)
    template_distances = torch.cdist(template[:1], template[:1])[0]
    all_atom_distances = torch.cdist(
        layout.template.coords[:1], layout.template.coords[:1]
    )[0]
    maximum_internal_error = 0.0
    for row in range(1, n_seeded + 1):
        torch.testing.assert_close(
            torch.cdist(placed[row : row + 1], placed[row : row + 1])[0],
            template_distances,
            rtol=0.0,
            # Internal geometry is preserved to the forward-kinematics fidelity,
            # not to fp64 machine precision: pose_stack_from_genes carries a
            # ~1e-6 A fp32 kinematic-rebuild artifact (F47), present for ANY
            # decode independent of the aligner. Placement (rigid frame) is exact
            # (< 1e-13, checked above); only the rebuilt internal geometry is
            # fp32-limited, which is negligible for docking.
            atol=1.0e-5,
        )
        row_error = torch.max(
            torch.abs(
                torch.cdist(
                    decoded.coords[row : row + 1], decoded.coords[row : row + 1]
                )[0]
                - all_atom_distances
            )
        )
        maximum_internal_error = max(maximum_internal_error, float(row_error))
    print(f"aligner seeded all-atom distance max error={maximum_internal_error:.3e} A")
    # ~1e-6 A fp32 forward-kinematics rebuild artifact (F47), not the aligner.
    assert maximum_internal_error <= 1.0e-5

    pairwise_placement_rmsd = torch.sqrt(
        torch.mean(
            torch.sum(
                (placed[1 : n_seeded + 1, None] - placed[None, 1 : n_seeded + 1])
                ** 2,
                dim=-1,
            ),
            dim=-1,
        )
    )
    off_diagonal = ~torch.eye(n_seeded, dtype=torch.bool)
    assert torch.all(pairwise_placement_rmsd[off_diagonal] >= 0.5 - 1.0e-12)


def test_refined_aligner_seeds_decode_to_selected_placements(cpu_aligner_case):
    case = cpu_aligner_case
    layout = _layout(case)
    aligner = LigandAligner(
        n_candidates=8,
        seed_fraction=0.25,
        cluster_rmsd=0.0,
        diversity_pool_size=4,
        score_batch_size=8,
        refinement_steps=2,
        refinement_top_k=2,
    )
    _, alignment, coordinate_error = _assert_seeded_coordinates_match_alignment(
        layout, case.backend, aligner, 1123, n_seeded=2
    )
    heavy_centroids = alignment.coordinates.index_select(
        1, layout.heavy_local_atoms
    ).mean(dim=1)
    torch.testing.assert_close(
        alignment.centroids, heavy_centroids, rtol=0.0, atol=2.0e-12
    )
    assert float(coordinate_error) <= 2.0e-12


def test_aligner_requires_grid_backend(cpu_aligner_case):
    case = cpu_aligner_case
    with pytest.raises(ValueError, match="requires a GridBackend"):
        run_ga_docking(
            case.reference,
            case.ligand_block,
            ContinuousBackend(case.param_db, kin_dtype=torch.float64),
            initialization="aligner",
        )


@pytest.mark.parametrize("npool,seed_fraction", [(1, 1.0), (4, 0.0), (4, 1.0)])
def test_aligner_seed_fraction_boundaries(cpu_aligner_case, npool, seed_fraction):
    case = cpu_aligner_case
    layout = _layout(case, npool=npool)
    aligner = LigandAligner(n_candidates=8, seed_fraction=seed_fraction)
    random, seeded = _initialize(layout, case.backend, aligner, 37)
    reference = layout.genes_from_pose_stack(layout.template)
    torch.testing.assert_close(seeded[0], reference[0], rtol=0.0, atol=0.0)
    if seed_fraction == 0.0:
        torch.testing.assert_close(seeded[1:], random[1:], rtol=0.0, atol=0.0)
    else:
        torch.testing.assert_close(
            seeded[:, 6:], reference[:, 6:], rtol=0.0, atol=2.0e-12
        )


def test_aligner_explicit_bounds_and_population_axis_rejection(cpu_aligner_case):
    case = cpu_aligner_case
    layout = _layout(case)
    aligner = LigandAligner(n_candidates=8)
    center = layout.ligand_heavy_coords(layout.template)[0].mean(dim=0)
    bounds = torch.stack((center - 1, center + 1))
    alignment = aligner.align(
        layout,
        case.backend,
        n_placements=4,
        pocket_bounds=bounds,
        generator=torch.Generator().manual_seed(41),
    )
    assert torch.all(alignment.centroids >= bounds[0])
    assert torch.all(alignment.centroids <= bounds[1])
    with pytest.raises(ValueError, match="shape"):
        aligner.align(
            layout,
            case.backend,
            n_placements=4,
            pocket_bounds=bounds[None],
            generator=torch.Generator().manual_seed(41),
        )


def test_aligner_ga_returns_one_ranked_population(cpu_aligner_case):
    case = cpu_aligner_case
    result = run_ga_docking(
        case.reference,
        case.ligand_block,
        case.backend.for_ligand_block(case.ligand_block),
        schedule=ga_module.GASchedule(generations=1, npool=3, maxiter=1),
        initialization="aligner",
        ligand_aligner=LigandAligner(n_candidates=8),
        seed=43,
    )
    assert result.pool.genes.shape[0] == 3
    assert result.ranked_energies.shape == (1, 3)
    assert result.best_pose_stack.n_poses == 1
    assert torch.all(torch.isfinite(result.pool.energies))
    assert torch.all(result.best_energy_history[1:] <= result.best_energy_history[:-1])


@pytest.fixture(scope="module")
def cpu_flexible_initialization_case(cpu_aligner_case):
    from tmol.tests.dock.test_flexible_sidechain_search import _cdk2_sidechain_blocks

    case = cpu_aligner_case
    sidechains = _cdk2_sidechain_blocks(case.reference, case.ligand_block)
    assert sidechains
    backend = GridBackend.from_pose_stack(
        case.reference,
        case.ligand_block,
        case.param_db,
        sidechain_blocks=sidechains,
        kin_dtype=torch.float64,
    )
    return SimpleNamespace(case=case, sidechains=sidechains, backend=backend)


@pytest.mark.parametrize("initialization", ["random", "conformer", "aligner"])
def test_flexible_initialization_retains_reference(
    cpu_flexible_initialization_case, monkeypatch, initialization
):
    prepared = cpu_flexible_initialization_case
    case, sidechains, backend = prepared.case, prepared.sidechains, prepared.backend
    captured = []
    original = ga_module.optimize_generation

    def capture(genes, layout, scoring_backend, **kwargs):
        reference = layout.genes_from_pose_stack(layout.template)
        captured.append((genes[0].clone(), reference[0].clone()))
        return original(genes, layout, scoring_backend, **kwargs)

    monkeypatch.setattr(ga_module, "optimize_generation", capture)
    run_ga_docking(
        case.reference,
        case.ligand_block,
        backend,
        sidechain_blocks=sidechains,
        schedule=ga_module.GASchedule(generations=1, npool=2, maxiter=1),
        initialization=initialization,
        ligand_aligner=LigandAligner(n_candidates=4),
        seed=47,
    )
    # Conformer without a mol2 also exercises reference retention on fallback.
    torch.testing.assert_close(captured[0][0], captured[0][1], rtol=0.0, atol=0.0)
