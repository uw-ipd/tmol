"""Correctness gates for grid-guided rigid-body ligand alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tmol.dock.ga as ga_module
from tmol.dock import (
    GridBackend,
    LigandAligner,
    LigandGeneLayout,
    RandomPlacement,
    build_ligand_docking_dofs,
    run_ga_docking,
)
from tmol.dock.ligand_aligner import (
    _axis_angle_rotation_matrix,
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
    refined = aligner._refine_candidates(QuadraticBackend(), coordinates, atom_indices)
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
    bounds = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64)
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
        torch.abs(decoded.coords[1 : n_seeded + 1] - expected_alignment.coordinates)
    )
    return genes, expected_alignment, coordinate_error


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
