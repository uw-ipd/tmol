"""Correctness gates for grid-guided rigid-body ligand alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tmol.dock.ga as ga_module
from tmol.dock import GridBackend, LigandAligner, RandomPlacement
from tmol.dock.ga import LigandGeneLayout, run_docking
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
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
        reference, param_db, ligand=ligand_block, kin_dtype=torch.float64
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
        case.param_db,
        ligand=case.ligand_block,
        flexible_residues=sidechains,
        kin_dtype=torch.float64,
    )
    return SimpleNamespace(case=case, sidechains=sidechains, backend=backend)


@pytest.mark.parametrize("initialization", ["random", "conformer", "aligner"])
def test_flexible_initialization_retains_reference(
    cpu_flexible_initialization_case, monkeypatch, initialization
):
    prepared = cpu_flexible_initialization_case
    case, backend = prepared.case, prepared.backend
    captured = []
    original = ga_module.optimize_generation

    def capture(genes, layout, scoring_backend, **kwargs):
        reference = layout.genes_from_pose_stack(layout.template)
        captured.append((genes[0].clone(), reference[0].clone()))
        return original(genes, layout, scoring_backend, **kwargs)

    monkeypatch.setattr(ga_module, "optimize_generation", capture)
    run_docking(
        case.reference,
        backend,
        schedule=ga_module.GASchedule(generations=1, npool=2, maxiter=1),
        initialization=initialization,
        ligand_aligner=LigandAligner(n_candidates=4),
        seed=47,
    )
    # Conformer without a mol2 also exercises reference retention on fallback.
    torch.testing.assert_close(captured[0][0], captured[0][1], rtol=0.0, atol=0.0)
