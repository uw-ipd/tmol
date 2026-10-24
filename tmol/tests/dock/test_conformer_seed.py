"""Tests for ETKDGv3/MMFF94 conformer-library GA initialization."""

from __future__ import annotations

import attr
import numpy as np
import pytest
import torch

import tmol.dock.ga as ga_module
from tmol.dock import (
    ContinuousBackend,
    GASchedule,
    LigandGeneLayout,
    build_ligand_docking_dofs,
    generate_conformer_library,
    run_ga_docking,
)
from tmol.tests.data import data_path
from tmol.tests.dock.test_ga import _TranslationImprovingBackend
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)
from tmol.pose import repeat_pose_stack

PLI_DATA_DIR = data_path("protein_ligand_test")
CDK2_MOL2 = PLI_DATA_DIR / "cdk2.lig.mol2"


@pytest.fixture(scope="module")
def cdk2_conformer_library():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    block_type = reference.block_type(0, ligand_block)
    return generate_conformer_library(
        CDK2_MOL2,
        block_type,
        reference.packed_block_types,
        n_conformers=20,
        energy_window=20.0,
        cluster_rmsd=0.5,
    )


def test_conformer_library_is_deterministic_and_identity_mapped(
    cdk2_conformer_library,
):
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    repeated = generate_conformer_library(
        CDK2_MOL2,
        reference.block_type(0, ligand_block),
        reference.packed_block_types,
        n_conformers=20,
        energy_window=20.0,
        cluster_rmsd=0.5,
    )

    library = cdk2_conformer_library
    assert library.atom_mapping_is_identity
    assert library.coords.shape[0] > 0
    assert library.coords.shape[1:] == (
        reference.block_type(0, ligand_block).n_atoms,
        3,
    )
    assert library.coords.dtype == np.float64
    assert np.max(library.energies) <= np.min(library.energies) + 20.0
    np.testing.assert_array_equal(library.coords, repeated.coords)
    np.testing.assert_array_equal(library.energies, repeated.energies)
    assert library.atom_mapping == repeated.atom_mapping


def test_conformer_seeded_genes_preserve_random_placement_and_valid_bonds(
    cdk2_conformer_library,
):
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    npool = 10
    population = repeat_pose_stack(reference, npool)
    layout = LigandGeneLayout.from_pose_stack(
        population,
        build_ligand_docking_dofs(population, ligand_block),
    )

    random = ga_module._random_initial_genes(
        layout,
        pocket_bounds=None,
        placement=ga_module.RandomPlacement(),
        generator=torch.Generator().manual_seed(73),
    )
    seeded = ga_module._conformer_initial_genes(
        layout,
        cdk2_conformer_library.coords,
        seed_fraction=0.5,
        pocket_bounds=None,
        placement=ga_module.RandomPlacement(),
        generator=torch.Generator().manual_seed(73),
    )
    torch.testing.assert_close(seeded[0], random[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(seeded[6:, 6:], random[6:, 6:], rtol=0.0, atol=0.0)
    assert torch.any(seeded[1:6, 6:] != random[1:6, 6:])

    expected_coords = layout.template.coords.clone()
    expected_rows = torch.arange(1, 6)
    expected_atoms = (
        layout.template.block_coord_offset64[expected_rows, ligand_block, None]
        + layout.ligand_local_atoms[None, :]
    )
    expected_library_rows = cdk2_conformer_library.coords[
        np.arange(expected_rows.shape[0]) % cdk2_conformer_library.coords.shape[0]
    ]
    expected_coords[expected_rows[:, None], expected_atoms] = torch.as_tensor(
        expected_library_rows,
        dtype=expected_coords.dtype,
    )
    expected_pose = attr.evolve(layout.template, coords=expected_coords)
    expected_genes = layout.genes_from_pose_stack(expected_pose)
    torch.testing.assert_close(
        seeded[1:6, 6:],
        expected_genes[1:6, 6:],
        rtol=0.0,
        atol=0.0,
    )

    random_pose = layout.pose_stack_from_genes(random)
    seeded_pose = layout.pose_stack_from_genes(seeded)
    torch.testing.assert_close(
        layout.ligand_heavy_coords(seeded_pose).mean(dim=1),
        layout.ligand_heavy_coords(random_pose).mean(dim=1),
        rtol=0.0,
        atol=1.0e-9,
    )

    block_type = reference.block_type(0, ligand_block)
    atom_index = block_type.atom_to_idx
    local_bonds = torch.tensor(
        [(atom_index[a], atom_index[b]) for a, b, *_ in block_type.bonds],
        dtype=torch.int64,
    )
    pose_indices = torch.arange(seeded_pose.n_poses)
    ligand_coords = seeded_pose.coords[
        pose_indices[:, None],
        seeded_pose.block_coord_offset64[:, ligand_block, None]
        + torch.arange(block_type.n_atoms)[None, :],
    ]
    bond_lengths = torch.linalg.vector_norm(
        ligand_coords[:, local_bonds[:, 0]] - ligand_coords[:, local_bonds[:, 1]],
        dim=2,
    )
    template_lengths = bond_lengths[0]
    torch.testing.assert_close(
        bond_lengths,
        template_lengths[None, :].expand_as(bond_lengths),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_conformer_ga_is_deterministic_and_builds_library_once(
    monkeypatch,
    cdk2_conformer_library,
):
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    calls = []

    def cached_library(*args, **kwargs):
        calls.append((args, kwargs))
        return cdk2_conformer_library

    monkeypatch.setattr(ga_module, "generate_conformer_library", cached_library)
    schedule = GASchedule(
        generations=1,
        npool=3,
        rmsdthreshold=0.0,
        pmut=0.2,
        maxiter=1,
    )
    result = run_ga_docking(
        reference,
        ligand_block,
        ContinuousBackend(param_db, kin_dtype=torch.float64),
        schedule=schedule,
        seed=79,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    )
    assert len(calls) == 1

    independent = run_ga_docking(
        reference,
        ligand_block,
        ContinuousBackend(param_db, kin_dtype=torch.float64),
        schedule=schedule,
        seed=79,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    )
    assert len(calls) == 2
    torch.testing.assert_close(
        result.best_genes,
        independent.best_genes,
        rtol=0.0,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        result.best_energies,
        independent.best_energies,
        rtol=0.0,
        atol=1.0e-9,
    )


def test_failed_conformer_generation_falls_back_byte_identically(monkeypatch, caplog):
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    schedule = GASchedule(
        generations=1,
        npool=3,
        rmsdthreshold=0.0,
        pmut=0.2,
        maxiter=1,
    )
    random = run_ga_docking(
        reference,
        ligand_block,
        _TranslationImprovingBackend(reference, ligand_block),
        schedule=schedule,
        seed=83,
        initialization="random",
    )

    def fail_generation(*args, **kwargs):
        raise ValueError("not MMFF-typeable")

    monkeypatch.setattr(ga_module, "generate_conformer_library", fail_generation)
    fallback = run_ga_docking(
        reference,
        ligand_block,
        _TranslationImprovingBackend(reference, ligand_block),
        schedule=schedule,
        seed=83,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    )
    assert torch.equal(fallback.pool.genes, random.pool.genes)
    assert torch.equal(fallback.pool.energies, random.pool.energies)
    assert "falling back to blind random initialization" in caplog.text
