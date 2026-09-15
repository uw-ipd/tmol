"""M8 gates for RosettaVS exact refinement and dG finalization."""

from __future__ import annotations

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.dock import (
    ContinuousBackend,
    GAPool,
    GAResult,
    GASchedule,
    GridBackend,
    count_non_proton_ligand_chis,
    run_docking,
)
from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.dock.finalization import (
    _promote_top_candidates,
    _receptor_only_pose_stack,
    finalize_ga_docking,
)
from tmol.dock.ga import _ligand_only_pose_stack
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.tests.dock.test_dock_loop import _load_cdk2_pose_on_device
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    PLI_DATA_DIR,
    _ligand_block,
    _load_cdk2_pose,
)
from tmol.tests.ligand.test_protein_ligand_ddg import _load_complex_cif


def _load_target(target: str, device: torch.device):
    pose_stack, context = pose_stack_from_biotite(
        _load_complex_cif(target),
        device,
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")],
        no_optH=True,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    return pose_stack, context.parameter_database


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
    ligand = _ligand_only_pose_stack(reference, ligand_block)

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
    population = _repeat_pose_stack(reference, 3)
    ligand_population = _ligand_only_pose_stack(population, ligand_block)
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
        finalize_ga_docking(population, ligand_block, None, ga_result, top_k=2)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.slow
def test_cdk2_grid_docking_exact_finalization_and_dg_assembly():
    cpu_reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(cpu_reference)
    grid_backend = GridBackend.from_pose_stack(
        cpu_reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float32,
    )
    reference, _ = _load_cdk2_pose_on_device(torch.device("cuda"))
    result = run_docking(
        reference,
        ligand_block,
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
    assert torch.all(final.dG[:, 1:] >= final.dG[:, :-1])
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


def test_finalize_false_does_not_call_finalization(monkeypatch):
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("finalization was called")

    monkeypatch.setattr("tmol.dock.finalization.finalize_ga_docking", fail_if_called)
    result = run_docking(
        reference,
        ligand_block,
        ContinuousBackend(param_db, kin_dtype=torch.float64),
        schedule=GASchedule(generations=1, npool=1, maxiter=1),
        seed=20260824,
        finalize=False,
    )
    assert result.finalization is None
    assert torch.equal(
        result.best_pose_stack.coords, result.ga_result.best_pose_stack.coords
    )
    assert torch.equal(result.best_energies, result.ga_result.best_energies)
    assert torch.equal(result.best_genes, result.ga_result.best_genes)
    assert torch.equal(result.ranked_energies, result.ga_result.ranked_energies)
    assert torch.equal(result.ranked_genes, result.ga_result.ranked_genes)
