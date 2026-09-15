"""M2e docking gates for the four-term rigid-receptor grid backend."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy
import pytest
import torch

import tmol.dock.ga as ga_module
import tmol.dock.scoring_backend as scoring_backend_module
from tmol.dock import (
    ContinuousBackend,
    GASchedule,
    GridBackend,
    LigandGeneLayout,
    build_ligand_docking_dofs,
    run_docking,
    run_ga_docking,
)
from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.dock.ga import _ligand_only_pose_stack
from tmol.optimization.minimizers import build_kinforest_network, run_min
from tmol.score.gridscore import score_ligand_grid_batched
from tmol.tests.dock.test_dock_loop import _load_cdk2_pose_on_device
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    _load_cdk2_pose,
)

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def cpu_grid_case():
    reference, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(reference)
    backend = GridBackend.from_pose_stack(
        reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float64,
    )
    population = _repeat_pose_stack(reference, 2)
    docking_dofs = build_ligand_docking_dofs(population, ligand_block)
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    return SimpleNamespace(
        reference=reference,
        param_db=param_db,
        ligand_block=ligand_block,
        backend=backend,
        population=population,
        docking_dofs=docking_dofs,
        layout=layout,
    )


@pytest.fixture(params=["full_complex", "ligand_only"])
def grid_pose_case(cpu_grid_case, request):
    case = cpu_grid_case
    if request.param == "full_complex":
        return case
    population = _ligand_only_pose_stack(case.population, case.ligand_block)
    docking_dofs = build_ligand_docking_dofs(population, 0)
    return SimpleNamespace(
        backend=case.backend.for_ligand_block(0),
        population=population,
        ligand_block=0,
        docking_dofs=docking_dofs,
        layout=LigandGeneLayout.from_pose_stack(population, docking_dofs),
    )


def test_grid_backend_score_matches_direct_batched_grid_score(cpu_grid_case):
    case = cpu_grid_case
    atom_indices = case.backend._ligand_atom_indices(
        case.population, case.ligand_block
    )
    pose_indices = torch.arange(case.population.n_poses, dtype=torch.int64)
    ligand_coordinates = case.population.coords[
        pose_indices[:, None], atom_indices
    ]
    expected = score_ligand_grid_batched(
        case.backend._grid_for(torch.device("cpu"), torch.float64),
        ligand_coordinates,
        case.backend.ligand_atom_types,
        case.backend._charges_for(torch.device("cpu"), torch.float64),
        score_function=case.backend._score_function(torch.device("cpu")),
    )
    actual = case.backend.score(case.population)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert case.backend._grid_cache[
        (torch.device("cpu"), torch.float64)
    ] is case.backend._grid_for(torch.device("cpu"), torch.float64)


def test_ligand_only_pose_stack_preserves_cdk2_ligand(cpu_grid_case):
    case = cpu_grid_case
    references = _repeat_pose_stack(case.reference, 2)
    source_indices = case.backend._ligand_atom_indices(references, case.ligand_block)
    references.coords[1, source_indices[1]] += 0.125
    ligand_only = _ligand_only_pose_stack(references, case.ligand_block)
    pose_indices = torch.arange(references.n_poses, dtype=torch.int64)

    assert ligand_only.n_poses == references.n_poses
    assert ligand_only.max_n_blocks == 1
    assert ligand_only.max_n_pose_atoms == len(case.backend.ligand_atom_types)
    assert ligand_only.inter_block_bondsep.shape[1:3] == (1, 1)
    assert torch.all(ligand_only.chain_id == 0)
    torch.testing.assert_close(
        ligand_only.coords,
        references.coords[pose_indices[:, None], source_indices],
        rtol=0.0,
        atol=0.0,
    )
    source_offset = int(references.block_coord_offset64[0, case.ligand_block])
    n_atoms = ligand_only.max_n_pose_atoms
    assert numpy.array_equal(
        ligand_only.pdb_info.atom_occupancy,
        references.pdb_info.atom_occupancy[:, source_offset : source_offset + n_atoms],
    )
    assert numpy.array_equal(
        ligand_only.pdb_info.atom_b_factor,
        references.pdb_info.atom_b_factor[:, source_offset : source_offset + n_atoms],
    )


def test_grid_ga_rejects_backend_for_another_ligand_block(cpu_grid_case):
    case = cpu_grid_case
    wrong_ligand_block = case.ligand_block - 1
    with pytest.raises(ValueError, match="backend was built for ligand block"):
        run_ga_docking(
            case.reference,
            wrong_ligand_block,
            case.backend,
            schedule=GASchedule(generations=1, npool=1, maxiter=1),
        )


class _FullComplexGridBackend:
    """Legacy full-complex adapter used as the M6 numerical oracle."""

    def __init__(self, backend):
        self.backend = backend

    def score(self, pose_stack):
        return self.backend.score(pose_stack)

    def minimize(self, pose_stack, docking_dofs, **kwargs):
        return self.backend.minimize(pose_stack, docking_dofs, **kwargs)


def _assert_ligand_only_grid_ga_matches_full_complex(
    reference, ligand_block, backend, atol
):
    schedule = GASchedule(
        generations=2,
        npool=4,
        rmsdthreshold=0.0,
        pmut=0.2,
        maxiter=5,
    )
    full_complex = run_ga_docking(
        reference,
        ligand_block,
        _FullComplexGridBackend(backend),
        schedule=schedule,
        seed=20260822,
    )
    ligand_only = run_ga_docking(
        reference,
        ligand_block,
        backend,
        schedule=schedule,
        seed=20260822,
    )

    history_err = float(
        torch.max(
            torch.abs(
                ligand_only.best_energy_history - full_complex.best_energy_history
            )
        )
    )
    gene_err = float(
        torch.max(torch.abs(ligand_only.best_genes - full_complex.best_genes))
    )
    coord_err = float(
        torch.max(
            torch.abs(
                ligand_only.best_pose_stack.coords - full_complex.best_pose_stack.coords
            )
        )
    )
    print(
        f"ligand-only vs full-complex grid GA ({ligand_only.best_genes.dtype}): "
        f"max |dHistory|={history_err:.3e} max |dGene|={gene_err:.3e} "
        f"max |dCoord|={coord_err:.3e} A (atol={atol:.1e})"
    )
    torch.testing.assert_close(
        ligand_only.best_energy_history,
        full_complex.best_energy_history,
        rtol=0.0,
        atol=atol,
    )
    torch.testing.assert_close(
        ligand_only.best_genes,
        full_complex.best_genes,
        rtol=0.0,
        atol=atol,
    )
    torch.testing.assert_close(
        ligand_only.best_pose_stack.coords,
        full_complex.best_pose_stack.coords,
        rtol=0.0,
        atol=atol,
    )
    assert ligand_only.pool.pose_stack.max_n_blocks == 1
    assert ligand_only.pool.pose_stack.max_n_pose_atoms == len(
        backend.ligand_atom_types
    )
    assert ligand_only.best_pose_stack.max_n_blocks == reference.max_n_blocks


@pytest.mark.slow
def test_ligand_only_grid_ga_matches_full_complex_cdk2_fp64(cpu_grid_case):
    # Definitive equivalence gate. In double precision the ligand-only and
    # full-complex representations run the identical ligand kinematics + grid
    # scoring and this reproduces the full GA trajectory to machine precision
    # (measured max |dHistory|=|dGene|=|dCoord|=0 in job 16255846).
    case = cpu_grid_case
    _assert_ligand_only_grid_ga_matches_full_complex(
        case.reference, case.ligand_block, case.backend, 1.0e-9
    )


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a CUDA device")
def test_ligand_only_grid_ga_matches_full_complex_cdk2_fp32(cpu_grid_case):
    # Single precision cannot reproduce the fp64 bit-equivalence above: the two
    # representations fold the ligand through different kinforest node orderings,
    # so fp32 reassociation alone perturbs coordinates by ~5e-4 A (measured max
    # |dCoord|=4.79e-4 A, |dGene|=1.5e-4, |dHistory|=4.6e-5 in job 16255847) --
    # negligible versus any docking-relevant scale. The tolerance reflects that
    # measured fp32 reassociation with margin; the fp64 test above is the actual
    # mathematical-equivalence gate. A GA selection flip would diverge by O(0.1-1
    # A) and still fail this bound loudly.
    case = cpu_grid_case
    reference, _ = _load_cdk2_pose_on_device(torch.device("cuda"))
    ligand_block = _ligand_block(reference)
    backend = GridBackend(
        case.backend.grids,
        ligand_block,
        case.backend.ligand_atom_types,
        case.backend.ligand_charges,
        case.param_db,
        kin_dtype=torch.float32,
    )
    _assert_ligand_only_grid_ga_matches_full_complex(
        reference, ligand_block, backend, 2.0e-3
    )


def test_grid_minimize_returns_direct_optimizer_gene(
    grid_pose_case, monkeypatch
):
    case = grid_pose_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[:, :6] += torch.tensor(
        [0.2, -0.1, 0.15, 0.08, -0.06, 0.04], dtype=genes.dtype
    )
    genes[:, 6:] += 0.2
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    proposed = case.layout.pose_stack_from_genes(genes)
    captured = {}

    def capture_noop_step(optimizer, closure):
        energies = closure()
        captured["parameter"] = optimizer._params[0]
        captured["energies"] = energies
        captured["gradient"] = optimizer._params[0].grad
        return energies.sum()

    monkeypatch.setattr(
        scoring_backend_module.LBFGS_Armijo, "step", capture_noop_step
    )
    result = case.backend.minimize(
        proposed,
        case.docking_dofs,
        maxiter=1,
        initial_raw_dofs=raw_dofs,
    )
    torch.testing.assert_close(
        result.masked_dofs,
        captured["parameter"].detach(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(result.genes, genes, rtol=0.0, atol=1.0e-12)
    assert captured["energies"].grad_fn is None
    assert captured["gradient"].grad_fn is None
    rebuilt = case.layout.pose_stack_from_genes(result.genes)
    torch.testing.assert_close(
        rebuilt.coords, result.pose_stack.coords, rtol=0.0, atol=1.0e-9
    )


def test_lightweight_grid_dof_gradient_matches_finite_difference(grid_pose_case):
    case = grid_pose_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[0, :6] += torch.tensor(
        [0.11, -0.07, 0.05, 0.03, -0.02, 0.04], dtype=genes.dtype
    )
    genes[1, :6] += torch.tensor(
        [-0.09, 0.08, -0.04, -0.02, 0.05, -0.03], dtype=genes.dtype
    )
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    proposed = case.layout.pose_stack_from_genes(genes)
    objective = scoring_backend_module._LightweightGridObjective(
        case.backend,
        proposed,
        case.docking_dofs,
        torch.float64,
        raw_dofs,
    )
    # Each representation uses its complete, valid kinforest, detached from
    # autograd. Only the full complex includes frozen receptor nodes.
    ligand_node_count = 1 + proposed.n_poses * len(case.backend.ligand_atom_types)
    if proposed.max_n_blocks == 1:
        assert objective.forest.id.numel() == ligand_node_count
    else:
        assert objective.forest.id.numel() > ligand_node_count
    masked = objective.initial_masked_dofs.clone().requires_grad_(True)
    energies, analytic = objective.energy_and_gradient(masked)
    finite = torch.empty_like(analytic)
    step = 1.0e-6
    for column in range(masked.numel()):
        plus = masked.detach().clone()
        minus = masked.detach().clone()
        plus[column] += step
        minus[column] -= step
        segment = int(objective.segment_ids[column])
        finite[column] = (
            objective.energy(plus)[segment]
            - objective.energy(minus)[segment]
        ) / (2.0 * step)

    max_error = torch.max(torch.abs(analytic - finite))
    print(f"grid freed-DOF gradient max error={float(max_error):.3e}")
    torch.testing.assert_close(analytic, finite, rtol=2.0e-5, atol=2.0e-5)
    assert energies.grad_fn is None
    assert analytic.grad_fn is None


@pytest.mark.slow
def test_lightweight_grid_minimize_matches_autograd_reference(grid_pose_case):
    case = grid_pose_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[:, :6] += torch.tensor(
        [0.2, -0.1, 0.15, 0.08, -0.06, 0.04], dtype=genes.dtype
    )
    genes[:, 6:] += 0.2
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    proposed = case.layout.pose_stack_from_genes(genes)

    reference_objective = scoring_backend_module._GridPoseObjective(
        case.backend, proposed, case.ligand_block
    )
    reference_network = build_kinforest_network(
        proposed,
        None,
        case.docking_dofs.fold_forest,
        case.docking_dofs.move_map,
        kin_dtype=torch.float64,
        whole_pose_scoring_module=reference_objective,
    )
    reference_network.full_dofs = raw_dofs.detach().clone()
    with torch.no_grad():
        reference_network.masked_dofs.copy_(
            reference_network.full_dofs[reference_network.dof_mask]
        )
    reference = run_min(
        reference_network, optimizer_kwargs={"max_iter": 5}
    )
    lightweight = case.backend.minimize(
        proposed,
        case.docking_dofs,
        maxiter=5,
        initial_raw_dofs=raw_dofs,
    )

    reference_energy = case.backend.score(reference)
    lightweight_energy = case.backend.score(lightweight.pose_stack)
    atom_indices = case.backend._ligand_atom_indices(
        proposed, case.ligand_block
    )
    pose_indices = torch.arange(proposed.n_poses, dtype=torch.int64)
    delta = (
        reference.coords[pose_indices[:, None], atom_indices]
        - lightweight.pose_stack.coords[pose_indices[:, None], atom_indices]
    )
    ligand_rms = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=2), dim=1))
    print(
        "lightweight-vs-autograd grid minimize: max |dE|="
        f"{float(torch.max(torch.abs(reference_energy-lightweight_energy))):.3e} "
        f"max ligand-RMS={float(torch.max(ligand_rms)):.3e} A"
    )
    torch.testing.assert_close(
        lightweight_energy, reference_energy, rtol=1.0e-7, atol=1.0e-7
    )
    torch.testing.assert_close(
        ligand_rms, torch.zeros_like(ligand_rms), rtol=0.0, atol=1.0e-7
    )


@pytest.mark.slow
def test_lightweight_grid_iteration_prefixes_are_monotone(cpu_grid_case):
    case = cpu_grid_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[:, :3] += torch.tensor([0.25, -0.15, 0.1], dtype=genes.dtype)
    pose_stack = case.layout.pose_stack_from_genes(genes)
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    energies = [case.backend.score(pose_stack)]
    for maxiter in range(1, 5):
        result = case.backend.minimize(
            pose_stack,
            case.docking_dofs,
            maxiter=maxiter,
            initial_raw_dofs=raw_dofs,
        )
        energies.append(case.backend.score(result.pose_stack))
    history = torch.stack(energies)
    torch.testing.assert_close(
        torch.minimum(history[1:], history[:-1]),
        history[1:],
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.slow
def test_grid_minimize_and_full_ga_cdk2_correctness(cpu_grid_case):
    case = cpu_grid_case
    genes = case.layout.genes_from_pose_stack(case.population)
    genes[1, 0] += 0.2
    genes[1, 1] -= 0.1
    proposed = case.layout.pose_stack_from_genes(genes)
    raw_dofs = case.layout.raw_dofs_from_genes(genes)
    before = case.backend.score(proposed)
    minimized = case.backend.minimize(
        proposed,
        case.docking_dofs,
        maxiter=5,
        initial_raw_dofs=raw_dofs,
    )
    after = case.backend.score(minimized.pose_stack)
    torch.testing.assert_close(
        torch.minimum(after, before), after, rtol=0.0, atol=1.0e-8
    )

    result = run_ga_docking(
        case.reference,
        case.ligand_block,
        case.backend,
        schedule=GASchedule(
            generations=2,
            npool=4,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=5,
        ),
        seed=20260822,
    )
    history = result.best_energy_history
    assert torch.all(history[1:] <= history[:-1] + 1.0e-8)
    assert torch.any(history[1:] < history[0] - 1.0e-8)
    assert torch.isfinite(result.pool.pose_stack.coords).all()

    native_indices = case.backend._ligand_atom_indices(
        case.reference, case.ligand_block
    )
    best_indices = case.backend._ligand_atom_indices(
        result.best_pose_stack, case.ligand_block
    )
    native = case.reference.coords[0, native_indices[0]]
    best = result.best_pose_stack.coords[0, best_indices[0]]
    atom_rms = torch.sqrt(torch.mean(torch.sum((best - native) ** 2, dim=1)))
    print(
        "cdk2 grid GA: history="
        f"{history.detach().cpu().tolist()} native-start atom-RMS="
        f"{float(atom_rms):.6f} A"
    )
    assert float(atom_rms) < 2.0


def test_docking_requires_one_reference_pose(cpu_grid_case):
    case = cpu_grid_case
    for runner in (run_ga_docking, run_docking):
        with pytest.raises(ValueError, match="must contain one reference pose"):
            runner(case.population, case.ligand_block, case.backend)
    with pytest.raises(ValueError, match="only a single reference pose"):
        _repeat_pose_stack(case.population, 2)


@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_random_initial_placement_is_seeded_and_bounded(
    cpu_grid_case, explicit_bounds
):
    case = cpu_grid_case
    population = _repeat_pose_stack(case.reference, 16)
    docking_dofs = build_ligand_docking_dofs(population, case.ligand_block)
    layout = LigandGeneLayout.from_pose_stack(population, docking_dofs)
    heavy = layout.ligand_heavy_coords(population)[0]
    placement = ga_module.RandomPlacement()
    if explicit_bounds:
        centroid = heavy.mean(dim=0)
        bounds = torch.stack((centroid - 1.0, centroid + 1.0))
    else:
        bounds = torch.stack(
            (
                heavy.amin(dim=0) - placement.pocket_padding_angstrom,
                heavy.amax(dim=0) + placement.pocket_padding_angstrom,
            )
        )

    def initialize():
        return ga_module._random_initial_genes(
            layout,
            pocket_bounds=bounds if explicit_bounds else None,
            placement=placement,
            generator=torch.Generator().manual_seed(91),
        )

    genes = initialize()
    torch.testing.assert_close(genes, initialize(), rtol=0.0, atol=0.0)
    assert torch.all(genes[:, 6:] > -math.pi)
    assert torch.all(genes[:, 6:] <= math.pi)
    placed = layout.pose_stack_from_genes(genes)
    placed_centroids = layout.ligand_heavy_coords(placed).mean(dim=1)
    assert torch.all(placed_centroids >= bounds[0] - 1.0e-9)
    assert torch.all(placed_centroids <= bounds[1] + 1.0e-9)


@pytest.mark.parametrize("backend_kind", ["continuous", "grid"])
@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires a CUDA device"
            ),
        ),
    ],
)
def test_run_docking_returns_ranked_single_cdk2_output(
    cpu_grid_case, backend_kind, device_name
):
    case = cpu_grid_case
    device = torch.device(device_name)
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    if device_name == "cpu":
        reference, param_db = case.reference, case.param_db
    else:
        reference, param_db = _load_cdk2_pose_on_device(device)
    ligand_block = _ligand_block(reference)
    if backend_kind == "continuous":
        backend = ContinuousBackend(param_db, kin_dtype=dtype)
    else:
        backend = GridBackend(
            case.backend.grids,
            ligand_block,
            case.backend.ligand_atom_types,
            case.backend.ligand_charges,
            param_db,
            kin_dtype=dtype,
        )
    schedule = GASchedule(
        generations=1,
        npool=2,
        rmsdthreshold=0.0,
        pmut=0.2,
        maxiter=1,
    )
    result = run_docking(
        reference,
        ligand_block,
        backend,
        schedule=schedule,
        seed=20260822,
    )

    assert result.best_pose_stack.n_poses == 1
    assert result.ga_result.pool.pose_stack.n_poses == schedule.npool
    assert result.best_energies.shape == (1,)
    assert result.best_genes.shape == (1, 8)
    assert result.ranked_energies.shape == (1, schedule.npool)
    assert result.ranked_genes.shape == (1, schedule.npool, 8)
    assert result.best_genes.device.type == device_name
    assert result.best_genes.dtype == dtype
    assert torch.isfinite(result.ranked_genes).all()
    assert torch.isfinite(result.ranked_energies).all()
    assert torch.isfinite(result.best_pose_stack.coords).all()
    assert torch.all(
        result.ranked_energies[:, 1:] >= result.ranked_energies[:, :-1]
    )
    torch.testing.assert_close(
        result.best_energies, result.ranked_energies[:, 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.best_genes, result.ranked_genes[:, 0], rtol=0.0, atol=0.0
    )
    best_index = torch.argsort(result.ga_result.pool.energies)[:1]
    expected_coords = result.ga_result.pool.pose_stack.coords[best_index]
    if backend_kind == "grid":
        assert result.ga_result.pool.pose_stack.max_n_blocks == 1
        assert expected_coords.shape[1] == len(backend.ligand_atom_types)
        ligand_indices = backend._ligand_atom_indices(reference, ligand_block)[0]
        complex_coords = reference.coords.clone()
        complex_coords[:, ligand_indices] = expected_coords
        expected_coords = complex_coords
    assert result.best_pose_stack.max_n_blocks == reference.max_n_blocks
    assert result.best_pose_stack.pdb_info is reference.pdb_info
    assert result.best_pose_stack.constraint_set is reference.constraint_set
    torch.testing.assert_close(
        result.best_pose_stack.coords,
        expected_coords,
        rtol=0.0,
        atol=0.0,
    )
