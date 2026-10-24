"""Correctness and derivative gates for live ligand-intra grid scoring."""

from __future__ import annotations

import attrs
import pytest
import torch

from tmol.dock import GridBackend
from tmol.score import beta2016_score_function
from tmol.score.gridscore import score_ligand_grid_batched
from tmol.score.gridscore.grid import score_ligand_grid_batched_with_gradients
from tmol.score.gridscore.live_ligand_intra import (
    build_live_ligand_intra_data,
    score_ligand_intra,
    score_ligand_intra_with_gradients,
)
from tmol.score import ScoreType
from tmol.tests.ligand.test_ligand_receptor_perterm_baseline import (
    _GOLDEN_PER_TERM,
    _TARGETS,
    _TERM_ORDER,
)
from tmol.tests.score.gridscore.test_gridscore import _load_target_pose
from tmol.tests.score.gridscore.test_live_terms import (
    _ligand_block,
    _ligand_coords,
)
from tmol.pose import repeat_pose_stack

_BASE_INTRA_TERMS = (
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "gen_torsions",
)
_CART_INTRA_TERMS = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)


def _perturbed_ligand_coords(pose_stack, ligand_block: int) -> torch.Tensor:
    base = _ligand_coords(pose_stack, ligand_block)[0]
    atom = torch.arange(base.shape[0], dtype=base.dtype)[:, None]
    axes = torch.arange(3, dtype=base.dtype)[None, :]
    delta = 0.035 * torch.sin(0.7 * atom + 1.3 * axes)
    return torch.stack((base, base + delta, base - 0.6 * delta))


def _full_complex_ligand_intra_oracle(
    pose_stack,
    ligand_block: int,
    param_db,
    ligand_coords: torch.Tensor,
    terms: tuple[str, ...],
) -> torch.Tensor:
    population = repeat_pose_stack(pose_stack, ligand_coords.shape[0])
    pose_indices = torch.arange(population.n_poses, dtype=torch.int64)
    type_index = int(population.block_type_ind64[0, ligand_block])
    n_atoms = population.packed_block_types.active_block_types[type_index].n_atoms
    atom_indices = (
        population.block_coord_offset64[:, ligand_block, None]
        + torch.arange(n_atoms, dtype=torch.int64)[None, :]
    )
    coords = population.coords.clone()
    coords[pose_indices[:, None], atom_indices] = ligand_coords
    population = attrs.evolve(population, coords=coords)
    score_function = beta2016_score_function(torch.device("cpu"), param_db=param_db)
    scorer = score_function.render_block_pair_scoring_module(population)
    per_term = scorer(population.coords, sum_terms=False)
    term_order = tuple(
        score_type.name for score_type in score_function.all_score_types()
    )
    assert term_order == _TERM_ORDER
    term_indices = [term_order.index(term) for term in terms]
    return per_term[term_indices, :, ligand_block, ligand_block].sum(dim=0)


@pytest.mark.parametrize("target", _TARGETS)
def test_full_live_ligand_intra_matches_exact_oracle_native_and_perturbed(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    data = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    ligand_coords = _perturbed_ligand_coords(pose_stack, ligand_block)
    # All five committed ligand parameter files contain bonds explicitly marked
    # as ring bonds, independently pinning the cart-bonded inclusion decision.
    assert data.include_cart_bonded
    terms = _BASE_INTRA_TERMS + _CART_INTRA_TERMS

    actual = score_ligand_intra(data, ligand_coords)
    padded = torch.empty((*ligand_coords.shape[:2], 6), dtype=ligand_coords.dtype)
    noncontiguous = padded[..., ::2]
    noncontiguous.copy_(ligand_coords)
    assert not noncontiguous.is_contiguous()
    noncontiguous_actual = score_ligand_intra(data, noncontiguous)
    expected = _full_complex_ligand_intra_oracle(
        pose_stack,
        ligand_block,
        param_db,
        ligand_coords,
        terms,
    )
    golden_native = sum(
        _GOLDEN_PER_TERM[target]["ligand_intra"][_TERM_ORDER.index(term)]
        for term in terms
    )
    oracle_error = torch.max(torch.abs(actual - expected))
    golden_error = abs(float(actual[0]) - golden_native)
    print(
        f"{target} ligand-intra: max live/exact error={float(oracle_error):.3e} "
        f"native golden error={golden_error:.3e}"
    )
    torch.testing.assert_close(noncontiguous_actual, actual, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(
        actual[0],
        torch.tensor(golden_native, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-4,
    )


@pytest.mark.parametrize("target", ("cdk2", "egfr"))
def test_live_ligand_intra_gradient_matches_autograd_and_finite_difference(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    data = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    coords = _perturbed_ligand_coords(pose_stack, ligand_block)[1:2].contiguous()

    energies, analytic = score_ligand_intra_with_gradients(data, coords)
    differentiable = coords.clone().requires_grad_(True)
    score_ligand_intra(data, differentiable).sum().backward()
    torch.testing.assert_close(analytic, differentiable.grad, rtol=0.0, atol=0.0)
    assert energies.grad_fn is None
    assert analytic.grad_fn is None

    step = 1.0e-6
    finite = torch.empty_like(coords)
    for column in range(coords.numel()):
        plus = coords.clone().reshape(-1)
        minus = coords.clone().reshape(-1)
        plus[column] += step
        minus[column] -= step
        finite.reshape(-1)[column] = (
            score_ligand_intra(data, plus.reshape_as(coords))[0]
            - score_ligand_intra(data, minus.reshape_as(coords))[0]
        ) / (2.0 * step)

    finite_error = torch.max(torch.abs(analytic - finite))
    print(
        f"{target} strained ligand-intra Cartesian gradient: "
        f"max error={float(finite_error):.3e}"
    )
    torch.testing.assert_close(analytic, finite, rtol=2.0e-5, atol=2.0e-5)


def test_ligand_intra_weight_profiles_match_gridscorer_mapping():
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    search = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="search"
    )
    full = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    beta = beta2016_score_function(torch.device("cpu"), param_db=param_db)

    assert search.include_cart_bonded
    assert full.include_cart_bonded
    assert float(search.score_function.get_weight(ScoreType.fa_ljrep)) == pytest.approx(
        0.01
    )
    assert float(
        search.score_function.get_weight(ScoreType.gen_torsions)
    ) == pytest.approx(0.1)
    for score_type in (ScoreType.fa_ljatr, ScoreType.fa_lk, ScoreType.fa_elec):
        assert float(search.score_function.get_weight(score_type)) == float(
            beta.get_weight(score_type)
        )
    assert float(full.score_function.get_weight(ScoreType.fa_ljrep)) == float(
        beta.get_weight(ScoreType.fa_ljrep)
    )
    if search.include_cart_bonded:
        for score_type in (
            ScoreType.cart_lengths,
            ScoreType.cart_angles,
            ScoreType.cart_torsions,
            ScoreType.cart_impropers,
            ScoreType.cart_hxltorsions,
        ):
            assert float(search.score_function.get_weight(score_type)) == pytest.approx(
                0.1
            )
            assert float(full.score_function.get_weight(score_type)) == float(
                beta.get_weight(score_type)
            )


@pytest.mark.slow
@pytest.mark.parametrize("target", ("cdk2", "hsp90"))
def test_disabled_ligand_intra_is_byte_identical_to_prior_grid_path(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    backend = GridBackend.from_pose_stack(
        pose_stack,
        ligand_block,
        param_db,
        kin_dtype=torch.float64,
        include_ligand_intra=False,
    )
    assert backend.live_ligand_intra is None
    assert backend._intra_for(torch.device("cpu"), torch.float64) is None

    coords = _perturbed_ligand_coords(pose_stack, ligand_block)[:2]
    direct_kwargs = dict(
        score_function=backend._score_function(torch.device("cpu")),
        live_hbond=backend._hbond_for(torch.device("cpu"), torch.float64),
        live_lk_ball=backend._lk_ball_for(torch.device("cpu"), torch.float64),
        lk_ball_grid=backend.lk_ball_grid,
    )
    legacy_energy = score_ligand_grid_batched(
        backend._grid_for(torch.device("cpu"), torch.float64),
        coords,
        backend.ligand_atom_types,
        backend._charges_for(torch.device("cpu"), torch.float64),
        **direct_kwargs,
    )
    disabled_energy = backend._score_ligand_coordinates(coords)
    torch.testing.assert_close(disabled_energy, legacy_energy, rtol=0.0, atol=0.0)

    legacy_with_gradients = score_ligand_grid_batched_with_gradients(
        backend._grid_for(torch.device("cpu"), torch.float64),
        coords,
        backend.ligand_atom_types,
        backend._charges_for(torch.device("cpu"), torch.float64),
        **direct_kwargs,
    )
    disabled_with_gradients = backend._score_ligand_coordinates_with_gradients(coords)
    torch.testing.assert_close(
        disabled_with_gradients[0], legacy_with_gradients[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        disabled_with_gradients[1], legacy_with_gradients[1], rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_live_ligand_intra_cuda_fp64_matches_cpu():
    pose_stack, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(pose_stack)
    cpu_data = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    cpu_coords = _perturbed_ligand_coords(pose_stack, ligand_block)
    expected = score_ligand_intra(cpu_data, cpu_coords)

    cuda_data = cpu_data.to(torch.device("cuda"), torch.float64)
    actual = score_ligand_intra(cuda_data, cpu_coords.to("cuda"))
    error = torch.max(torch.abs(actual.cpu() - expected))
    print(f"CUDA fp64 ligand-intra max CPU error={float(error):.3e}")
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_live_ligand_intra_cuda_fp64_gradient_matches_cpu():
    """Directly exercise the CUDA backward: CUDA analytic gradient == CPU's.

    The CPU analytic gradient is finite-difference-validated elsewhere, so
    matching it on CUDA establishes CUDA backward correctness (which the
    ligand-only-vs-full-complex GA parity test cannot, as both paths share the
    same intra-gradient implementation).
    """
    pose_stack, param_db = _load_target_pose("egfr")
    ligand_block = _ligand_block(pose_stack)
    cpu_data = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    cpu_coords = _perturbed_ligand_coords(pose_stack, ligand_block)[1:2].contiguous()
    cpu_e, cpu_g = score_ligand_intra_with_gradients(cpu_data, cpu_coords)

    cuda_data = cpu_data.to(torch.device("cuda"), torch.float64)
    cuda_e, cuda_g = score_ligand_intra_with_gradients(cuda_data, cpu_coords.to("cuda"))
    e_err = float(torch.max(torch.abs(cuda_e.cpu() - cpu_e)))
    g_err = float(torch.max(torch.abs(cuda_g.cpu() - cpu_g)))
    print(
        f"CUDA vs CPU ligand-intra gradient: max dE={e_err:.3e} max dGrad={g_err:.3e}"
    )
    assert cuda_g.grad_fn is None
    torch.testing.assert_close(cuda_e.cpu(), cpu_e, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(cuda_g.cpu(), cpu_g, rtol=0.0, atol=1.0e-6)
