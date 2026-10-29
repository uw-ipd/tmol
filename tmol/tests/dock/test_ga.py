"""M4 gates for vectorized, device-resident GALigandDock operators."""

from __future__ import annotations


import attr
import pytest
import torch


import tmol.dock.ga as ga_module
from tmol.dock import ContinuousBackend, GASchedule
from tmol.dock._diversity import ligand_heavy_atom_rms
from tmol.dock.ga import LigandGeneLayout, run_docking
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.dock.scoring_backend import MinimizationResult, ScoringBackend
from tmol.tests.dock.test_ligand_dof_roundtrip import _load_cdk2_pose_on_device
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)

CUDA_AVAILABLE = torch.cuda.is_available()


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
    parents = torch.tensor([[0.0, 0.0, 0.0], [0.23, -0.31, 0.47]], dtype=dtype)
    axes = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=dtype)
    axes = axes / torch.linalg.vector_norm(axes, dim=1, keepdim=True)
    angles = torch.tensor([0.19, -0.37], dtype=dtype)

    mutated_euler = ga_module._apply_rosetta_rotation_mutation(parents, axes, angles)
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


class _TranslationImprovingBackend(ScoringBackend):
    def __init__(self, reference, ligand_block):
        self.ligand_block = ligand_block
        block_type_index = int(reference.block_type_ind64[0, ligand_block])
        block_type = reference.packed_block_types.active_block_types[block_type_index]
        self.offset = int(reference.block_coord_offset64[0, ligand_block])
        self.n_atoms = block_type.n_atoms
        ligand = reference.coords[0, self.offset : self.offset + self.n_atoms]
        self.target_com = torch.mean(ligand, dim=0) + torch.tensor(
            [10.0, 0.0, 0.0], dtype=ligand.dtype, device=ligand.device
        )
        self.maxiters = []
        self.population_sizes = []
        self.proposed_genes = []
        self.minimized_genes = []

    def score(self, pose_stack):
        ligand = pose_stack.coords[:, self.offset : self.offset + self.n_atoms]
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
        coords[:, self.offset : self.offset + self.n_atoms] = ligand + shift[:, None, :]
        layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
        genes = layout.genes_from_raw_dofs(initial_raw_dofs).clone()
        self.proposed_genes.append(genes.detach().clone())
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
    result = run_docking(
        reference, backend, schedule=schedule, seed=7, initialization="jitter"
    ).ga_result

    history = result.best_energy_history
    assert result.pool.genes.shape == (schedule.npool, 8)
    assert torch.all(history[1:] <= history[:-1])
    assert torch.any(history[1:] < history[0])
    assert backend.maxiters == [schedule.maxiter] * (schedule.generations + 1)


def test_cpu_fp64_continuous_ga_smoke():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    result = run_docking(
        reference,
        ContinuousBackend(ligand_block, param_db, kin_dtype=torch.float64),
        schedule=GASchedule(
            generations=1,
            npool=2,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=2,
        ),
        seed=20260821,
        initialization="jitter",
    ).ga_result
    assert result.pool.genes.shape == (2, 8)
    assert result.pool.genes.dtype == torch.float64
    assert torch.isfinite(result.pool.genes).all()
    assert torch.isfinite(result.pool.energies).all()
    assert torch.all(result.best_energy_history[1:] <= result.best_energy_history[:-1])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
def test_gpu_fp32_cdk2_default_vsx_schedule():
    device = torch.device("cuda")
    reference, param_db = _load_cdk2_pose_on_device(device)
    ligand_block = _ligand_block(reference)
    schedule = GASchedule()
    result = run_docking(
        reference,
        ContinuousBackend(ligand_block, param_db, kin_dtype=torch.float32),
        schedule=schedule,
        seed=20260821,
        initialization="jitter",
    ).ga_result

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
