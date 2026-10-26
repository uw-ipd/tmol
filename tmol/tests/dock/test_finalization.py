"""M8 gates for RosettaVS exact refinement and dG finalization."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

import tmol.dock.finalization as finalization_module
from tmol.database import ParameterDatabase
from tmol.dock import GASchedule, GridBackend, run_docking, select_pocket_residues
from tmol.dock.finalization import count_non_proton_ligand_chis
from tmol.dock.ga import GAPool, GAResult, LigandGeneLayout
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.dock.finalization import (
    _flexible_sidechain_mapping,
    _ligand_atom_indices,
    _mapped_atom_indices,
    _promote_top_candidates,
    _receptor_only_pose_stack,
    _score_pose_stack,
    finalize_ga_docking,
)
from tmol.io import pose_stack_from_biotite
from tmol.score import beta2016_score_function
from tmol.tests.dock.test_ligand_dof_roundtrip import _load_cdk2_pose_on_device
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    PLI_DATA_DIR,
    _ligand_block,
    _load_cdk2_pose,
)
from tmol.tests.ligand.test_protein_ligand_ddg import _load_complex_cif
from tmol.pose import repeat_pose_stack
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_pose_stack,
    sidechain_context_blocks,
)


def _load_target(target: str, device: torch.device):
    pose_stack, context = pose_stack_from_biotite(
        _load_complex_cif(target),
        device,
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    return pose_stack, context.parameter_database


def _flexible_sidechains(reference, ligand_block, count=1):
    selected = select_pocket_residues(reference, ligand_block, cutoff=6.0)
    sidechain_blocks = tuple(
        block
        for block in selected.detach().cpu().tolist()
        if reference.block_type(0, block).n_sc_torsions
    )
    assert len(sidechain_blocks) >= count
    return sidechain_blocks[:count]


def _flexible_ga_result(reference, ligand_block, sidechain_blocks, npool=3):
    movable = movable_pose_stack(reference, ligand_block, sidechain_blocks)
    _, compact_sidechains, _ = movable_block_mapping(
        reference, ligand_block, sidechain_blocks
    )
    population = repeat_pose_stack(movable, npool)
    docking_dofs = build_ligand_docking_dofs(
        population, 0, sidechain_blocks=compact_sidechains
    )
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    genes = layout.genes_from_pose_stack(population)
    first_sidechain_gene = 6 + len(docking_dofs.torsions)
    per_pose_increment = torch.linspace(
        0.15,
        0.15 * npool,
        npool,
        dtype=genes.dtype,
        device=genes.device,
    )
    per_chi_scale = torch.arange(
        1,
        len(docking_dofs.sidechain_torsions) + 1,
        dtype=genes.dtype,
        device=genes.device,
    )
    genes[:, first_sidechain_gene:] += (
        per_pose_increment[:, None] * per_chi_scale[None, :]
    )
    population = layout.pose_stack_from_genes(genes)
    energies = torch.arange(npool, 0, -1, dtype=genes.dtype, device=genes.device)
    if npool == 3:
        energies = torch.tensor([3.0, 1.0, 2.0], dtype=genes.dtype, device=genes.device)
    order = torch.argsort(energies)
    return GAResult(
        pool=GAPool(genes, population, energies, n_diverse=npool),
        best_energy_history=energies[order[:1]],
        diverse_count_history=(),
        ranked_energies=energies[order][None, :],
        ranked_genes=genes[order][None, :, :],
    )


def _bad_bond_count(reference, actual, blocks):
    bad_bonds = 0
    for block in blocks:
        block_type = reference.block_type(0, block)
        bonds = torch.tensor(
            block_type.bond_indices[::2],
            dtype=torch.int64,
            device=reference.device,
        )
        offset = reference.block_coord_offset64[0, block]
        lengths = torch.linalg.vector_norm(
            actual.coords[0, offset + bonds[:, 0]]
            - actual.coords[0, offset + bonds[:, 1]],
            dim=1,
        )
        bad_bonds += int(torch.sum((lengths < 0.8) | (lengths > 2.0)))
    return bad_bonds


def test_simple_entropy_excludes_known_ace_proton_chi():
    pose_stack, _ = _load_target("ace", torch.device("cpu"))
    ligand_block = _ligand_block(pose_stack)
    block_type = pose_stack.block_type(0, ligand_block)

    # The pinned ACE tmol residue declares 13 chis; chi13 is N3-H2.
    assert block_type.n_torsions == 13
    assert count_non_proton_ligand_chis(pose_stack, ligand_block) == 12


def test_receptor_only_builder_removes_ligand_and_preserves_receptor():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    receptor = _receptor_only_pose_stack(reference, ligand_block)
    ligand = movable_pose_stack(reference, ligand_block)

    assert receptor.max_n_blocks == reference.max_n_blocks - 1
    assert receptor.max_n_pose_atoms + ligand.max_n_pose_atoms == (
        int(reference.n_ats_per_block[0].sum())
    )
    torch.testing.assert_close(
        receptor.coords,
        reference.coords[:, : receptor.max_n_pose_atoms],
        rtol=0.0,
        atol=0.0,
    )


def test_top_candidates_are_promoted_without_moving_receptor():
    reference, _ = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    population = repeat_pose_stack(reference, 3)
    ligand_population = movable_pose_stack(population, ligand_block)
    ligand_population.coords[:, :, 0] += torch.tensor([1.0, 2.0, 3.0])[:, None]
    energies = torch.tensor([3.0, 1.0, 2.0], dtype=reference.coords.dtype)
    genes = torch.arange(24, dtype=reference.coords.dtype).reshape(3, 8)
    pool = GAPool(genes, ligand_population, energies, n_diverse=3)
    ga_result = GAResult(
        pool=pool,
        best_energy_history=torch.tensor([1.0]),
        diverse_count_history=(),
        ranked_energies=torch.sort(energies)[0][None, :],
        ranked_genes=genes[torch.argsort(energies)][None, :, :],
    )

    with pytest.raises(ValueError, match="requires one reference pose"):
        _promote_top_candidates(population, ligand_block, ga_result, top_k=2)
    with pytest.raises(ValueError, match="requires one reference pose"):
        finalize_ga_docking(population, None, ga_result, top_k=2)

    promoted, promoted_genes = _promote_top_candidates(
        reference, ligand_block, ga_result, top_k=2
    )
    ligand_indices = (
        promoted.block_coord_offset64[:, ligand_block, None]
        + torch.arange(ligand_population.max_n_pose_atoms)[None, :]
    )
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(
        promoted.coords[rows, ligand_indices],
        ligand_population.coords[[1, 2]],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(promoted_genes[0], genes[[1, 2]])
    receptor_atoms = torch.arange(int(reference.block_coord_offset64[0, ligand_block]))
    torch.testing.assert_close(
        promoted.coords[:, receptor_atoms],
        reference.coords[:, receptor_atoms].expand(2, -1, -1),
        rtol=0.0,
        atol=0.0,
    )


def test_flexible_top_candidates_restore_distinct_sidechains():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _flexible_sidechains(reference, ligand_block, count=2)
    ga_result = _flexible_ga_result(reference, ligand_block, sidechain_blocks, npool=3)
    backend = SimpleNamespace(
        param_db=param_db,
        ligand_block=ligand_block,
        sidechain_blocks=sidechain_blocks,
        context_blocks=sidechain_context_blocks(
            reference, ligand_block, sidechain_blocks
        ),
    )
    (
        _,
        compact_sidechains,
        compact_atoms,
        block_local_atoms,
    ) = _flexible_sidechain_mapping(
        reference, ligand_block, backend, ga_result.pool.pose_stack
    )

    promoted, promoted_genes = _promote_top_candidates(
        reference,
        ligand_block,
        ga_result,
        top_k=2,
        sidechain_blocks=sidechain_blocks,
        compact_sidechain_blocks=compact_sidechains,
        sidechain_atom_indices=compact_atoms,
    )
    full_atoms = _mapped_atom_indices(promoted, block_local_atoms)
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(
        promoted.coords[rows, full_atoms],
        ga_result.pool.pose_stack.coords[[1, 2]][:, compact_atoms],
        rtol=0.0,
        atol=1.0e-9,
    )
    torch.testing.assert_close(promoted_genes[0], ga_result.pool.genes[[1, 2]])
    assert torch.any(
        promoted.coords[0, full_atoms[0]] != reference.coords[0, full_atoms[0]]
    )
    assert torch.any(
        promoted.coords[0, full_atoms[0]] != promoted.coords[1, full_atoms[1]]
    )

    promoted_one, _ = _promote_top_candidates(
        reference,
        ligand_block,
        ga_result,
        top_k=1,
        sidechain_blocks=sidechain_blocks,
        compact_sidechain_blocks=compact_sidechains,
        sidechain_atom_indices=compact_atoms,
    )
    one_full_atoms = _mapped_atom_indices(promoted_one, block_local_atoms)
    torch.testing.assert_close(
        promoted_one.coords[0, one_full_atoms[0]],
        ga_result.pool.pose_stack.coords[1, compact_atoms],
        rtol=0.0,
        atol=1.0e-9,
    )


@pytest.mark.slow
def test_cdk2_cpu_flexible_finalization_refines_sidechain_and_apo(monkeypatch):
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _flexible_sidechains(reference, ligand_block)
    ga_result = _flexible_ga_result(reference, ligand_block, sidechain_blocks, npool=1)
    backend = SimpleNamespace(
        param_db=param_db,
        ligand_block=ligand_block,
        sidechain_blocks=sidechain_blocks,
        context_blocks=sidechain_context_blocks(
            reference, ligand_block, sidechain_blocks
        ),
    )
    (
        _,
        compact_sidechains,
        compact_atoms,
        block_local_atoms,
    ) = _flexible_sidechain_mapping(
        reference, ligand_block, backend, ga_result.pool.pose_stack
    )
    promoted, _ = _promote_top_candidates(
        reference,
        ligand_block,
        ga_result,
        top_k=1,
        sidechain_blocks=sidechain_blocks,
        compact_sidechain_blocks=compact_sidechains,
        sidechain_atom_indices=compact_atoms,
    )
    promoted_sidechain_atoms = _mapped_atom_indices(promoted, block_local_atoms)

    captured_apo = {}
    real_run_cart_min = finalization_module.run_cart_min

    def capture_apo(*args, **kwargs):
        result = real_run_cart_min(*args, **kwargs)
        if args[0].max_n_blocks == reference.max_n_blocks - 1:
            captured_apo["before"] = args[0]
            captured_apo["after"] = result
            captured_apo["coord_mask"] = kwargs["coord_mask"]
        return result

    monkeypatch.setattr(finalization_module, "run_cart_min", capture_apo)
    final = finalize_ga_docking(reference, backend, ga_result, top_k=1)
    final_sidechain_atoms = _mapped_atom_indices(final.pose_stack, block_local_atoms)
    sidechain_refine_delta = torch.amax(
        torch.linalg.vector_norm(
            final.pose_stack.coords[0, final_sidechain_atoms[0]]
            - promoted.coords[0, promoted_sidechain_atoms[0]],
            dim=1,
        )
    )
    assert sidechain_refine_delta > 1.0e-6

    moving_mask = torch.zeros(
        reference.max_n_pose_atoms, dtype=torch.bool, device=reference.device
    )
    moving_mask[_ligand_atom_indices(reference, ligand_block)[0]] = True
    moving_mask[final_sidechain_atoms[0]] = True
    torch.testing.assert_close(
        final.pose_stack.coords[0, ~moving_mask],
        reference.coords[0, ~moving_mask],
        rtol=0.0,
        atol=1.0e-6,
    )
    assert (
        _bad_bond_count(
            reference,
            final.pose_stack,
            (ligand_block,) + sidechain_blocks,
        )
        == 0
    )
    assert torch.isfinite(final.complex_energies).all()
    assert torch.all(final.complex_energies <= final.pre_refine_energies + 1.0e-8)

    exact_sfxn = beta2016_score_function(reference.device, param_db=param_db)
    raw_receptor_energy = _score_pose_stack(
        _receptor_only_pose_stack(reference, ligand_block).split(0), exact_sfxn
    )[0]
    assert final.free_receptor_energies[0, 0] <= raw_receptor_energy + 1.0e-8
    assert not torch.equal(final.free_receptor_energies[0, 0], raw_receptor_energy)
    apo_before = captured_apo["before"]
    apo_after = captured_apo["after"]
    apo_mask = captured_apo["coord_mask"][0]
    torch.testing.assert_close(
        apo_after.coords[0, ~apo_mask],
        apo_before.coords[0, ~apo_mask],
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.any(
        torch.linalg.vector_norm(
            apo_after.coords[0, apo_mask] - apo_before.coords[0, apo_mask], dim=1
        )
        > 1.0e-6
    )
    real_blocks = torch.nonzero(
        reference.block_type_ind64[0] >= 0, as_tuple=False
    ).flatten()
    keep_blocks = real_blocks[real_blocks != ligand_block].tolist()
    source_to_receptor = {
        source_block: receptor_block
        for receptor_block, source_block in enumerate(keep_blocks)
    }
    assert (
        _bad_bond_count(
            apo_before,
            apo_after,
            tuple(source_to_receptor[block] for block in sidechain_blocks),
        )
        == 0
    )
    torch.testing.assert_close(
        final.dH,
        final.complex_energies
        - final.free_ligand_energies
        - final.free_receptor_energies,
    )
    print(
        "cpu fp64 flexible sidechain refine delta, apo raw/relaxed/delta, dH:",
        float(sidechain_refine_delta),
        float(raw_receptor_energy),
        float(final.free_receptor_energies[0, 0]),
        float(final.free_receptor_energies[0, 0] - raw_receptor_energy),
        float(final.dH[0, 0]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.slow
def test_cdk2_grid_docking_exact_finalization_and_dg_assembly():
    cpu_reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(cpu_reference)
    grid_backend = GridBackend.from_pose_stack(
        cpu_reference, ligand_block, param_db, kin_dtype=torch.float32
    )
    reference, _ = _load_cdk2_pose_on_device(torch.device("cuda"))
    result = run_docking(
        reference,
        grid_backend,
        schedule=GASchedule(
            generations=1,
            npool=2,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=1,
        ),
        seed=20260824,
        finalize=True,
        top_k=2,
    )
    final = result.finalization
    assert final is not None
    assert final.pose_stack.n_poses == 2
    assert torch.isfinite(final.pose_stack.coords).all()
    assert torch.isfinite(final.dG).all()
    assert torch.all(final.complex_energies <= final.pre_refine_energies + 5.0e-3)
    assert torch.any(final.complex_energies < final.pre_refine_energies)
    torch.testing.assert_close(
        final.dH,
        final.complex_energies
        - final.free_ligand_energies
        - final.free_receptor_energies,
    )
    torch.testing.assert_close(final.TdS, 0.4 * final.nchi.to(final.TdS.dtype))
    torch.testing.assert_close(final.dG, final.dH + final.TdS)
    torch.testing.assert_close(
        final.free_receptor_energies[:, 1:],
        final.free_receptor_energies[:, :1],
        rtol=0.0,
        atol=0.0,
    )
    # Default top_pose_metric="score" sorts candidates by exact complex energy.
    assert torch.all(final.complex_energies[:, 1:] >= final.complex_energies[:, :-1])
    torch.testing.assert_close(result.ranked_energies, final.dG)
    torch.testing.assert_close(result.dH, final.dH)
    torch.testing.assert_close(result.TdS, final.TdS)
    torch.testing.assert_close(result.dG, final.dG)

    receptor_atoms = torch.arange(
        int(reference.block_coord_offset64[0, ligand_block]),
        device=reference.device,
    )
    torch.testing.assert_close(
        final.pose_stack.coords[:, receptor_atoms],
        reference.coords[:, receptor_atoms].expand(2, -1, -1),
        rtol=0.0,
        atol=0.0,
    )
    print(
        "cdk2 exact beta2016 pre/post:",
        final.pre_refine_energies.detach().cpu().tolist(),
        final.complex_energies.detach().cpu().tolist(),
        "dH/TdS/dG:",
        final.dH.detach().cpu().tolist(),
        final.TdS.detach().cpu().tolist(),
        final.dG.detach().cpu().tolist(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.slow
def test_cdk2_flexible_grid_docking_exact_finalization_e2e():
    cpu_reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(cpu_reference)
    sidechain_blocks = _flexible_sidechains(cpu_reference, ligand_block)
    grid_backend = GridBackend.from_pose_stack(
        cpu_reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float32,
        flexible_residues=sidechain_blocks,
    )
    reference, _ = _load_cdk2_pose_on_device(torch.device("cuda"))
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run_docking(
        reference,
        grid_backend,
        schedule=GASchedule(
            generations=1,
            npool=2,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=1,
        ),
        seed=20260826,
        finalize=True,
        top_k=2,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    final = result.finalization
    assert final is not None
    assert torch.isfinite(final.dG).all()
    assert torch.all(final.complex_energies <= final.pre_refine_energies + 5.0e-3)
    _, _, _, block_local_atoms = _flexible_sidechain_mapping(
        reference,
        ligand_block,
        grid_backend,
        result.ga_result.pool.pose_stack,
    )
    sidechain_atoms = _mapped_atom_indices(final.pose_stack, block_local_atoms)
    displacement = torch.linalg.vector_norm(
        final.pose_stack.coords[0, sidechain_atoms[0]]
        - reference.coords[0, sidechain_atoms[0]],
        dim=1,
    )
    assert torch.any(displacement > 1.0e-5)

    moving_mask = torch.zeros(
        reference.max_n_pose_atoms, dtype=torch.bool, device=reference.device
    )
    moving_mask[_ligand_atom_indices(reference, ligand_block)[0]] = True
    moving_mask[sidechain_atoms[0]] = True
    torch.testing.assert_close(
        final.pose_stack.coords[0, ~moving_mask],
        reference.coords[0, ~moving_mask],
        rtol=0.0,
        atol=1.0e-6,
    )
    bad_bonds = _bad_bond_count(
        reference,
        final.pose_stack,
        (ligand_block,) + sidechain_blocks,
    )
    ligand_atoms = _ligand_atom_indices(reference, ligand_block)[0]
    rmsd_to_native = torch.sqrt(
        torch.mean(
            torch.sum(
                (
                    final.pose_stack.coords[0, ligand_atoms]
                    - reference.coords[0, ligand_atoms]
                )
                ** 2,
                dim=1,
            )
        )
    )
    print(
        "flexible cdk2 finalized RMSD, wall_s, moved_sidechain_atoms, " "bad_bonds:",
        float(rmsd_to_native),
        wall_seconds,
        int(torch.sum(displacement > 1.0e-5)),
        bad_bonds,
    )
    assert bad_bonds == 0
