"""Tests for ETKDGv3/MMFF94 conformer-library GA initialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import tmol.dock.ga as ga_module
from tmol.dock import ContinuousBackend, GASchedule
from tmol.dock.conformer_seed import generate_conformer_library
from tmol.dock.ga import run_docking
from tmol.tests.data import data_path
from tmol.tests.dock.test_ga import _TranslationImprovingBackend
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)

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
    result = run_docking(
        reference,
        ContinuousBackend(ligand_block, param_db, kin_dtype=torch.float64),
        schedule=schedule,
        seed=79,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    ).ga_result
    assert len(calls) == 1

    independent = run_docking(
        reference,
        ContinuousBackend(ligand_block, param_db, kin_dtype=torch.float64),
        schedule=schedule,
        seed=79,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    ).ga_result
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
    random = run_docking(
        reference,
        _TranslationImprovingBackend(reference, ligand_block),
        schedule=schedule,
        seed=83,
        initialization="random",
    ).ga_result

    def fail_generation(*args, **kwargs):
        raise ValueError("not MMFF-typeable")

    monkeypatch.setattr(ga_module, "generate_conformer_library", fail_generation)
    fallback = run_docking(
        reference,
        _TranslationImprovingBackend(reference, ligand_block),
        schedule=schedule,
        seed=83,
        initialization="conformer",
        ligand_mol2=CDK2_MOL2,
    ).ga_result
    assert torch.equal(fallback.pool.genes, random.pool.genes)
    assert torch.equal(fallback.pool.energies, random.pool.energies)
    assert "falling back to blind random initialization" in caplog.text


def test_atom_mapping_falls_back_to_graph_match_for_symmetric_ligand():
    from rdkit import Chem

    from tmol.dock.conformer_seed import _block_atom_mapping, _mapping_is_valid

    reference, _ = _load_cdk2_pose()
    block_type = reference.block_type(0, _ligand_block(reference))
    pbt = reference.packed_block_types
    mol = Chem.MolFromMol2File(str(CDK2_MOL2), removeHs=False)
    n_atoms = mol.GetNumAtoms()
    # Reversed atom order with names that match nothing: neither the name-based
    # nor the identity mapping is valid, and cdk2's symmetric aryl ring makes the
    # graph isomorphism ambiguous. Any isomorphism must be accepted.
    mol = Chem.RenumberAtoms(mol, list(reversed(range(n_atoms))))
    names = tuple(f"X{i}" for i in range(n_atoms))
    mapping = _block_atom_mapping(mol, names, block_type, pbt)
    assert sorted(mapping) == list(range(n_atoms))
    assert _mapping_is_valid(mol, block_type, pbt, mapping)
