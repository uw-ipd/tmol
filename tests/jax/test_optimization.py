"""Cartesian minimization must decrease energy and honor exact movement masks."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax import beta2016_score_function, pose_stack_from_pdb
from tmol_jax.optimization import lbfgs_minimize, run_cart_min, run_cart_relax
from tmol_jax.score import score_function as scoring
from tmol_jax.score import ScoreFunction, ScoreType
from tmol_jax.score.constraint import ConstraintSet


def test_lbfgs_quadratic_and_mask():
    target = jnp.asarray([2.0, -3.0, 4.0])

    def objective(x):
        return jnp.sum((x - target) ** 2)

    result = lbfgs_minimize(
        objective,
        jnp.asarray([0.0, 5.0, 9.0]),
        grad_mask=jnp.asarray([1.0, 0.0, 1.0]),
    )
    assert result["converged"]
    result_x = np.asarray(result["x"])
    target_x = np.asarray(target)
    np.testing.assert_allclose(result_x[[0, 2]], target_x[[0, 2]], atol=1e-5)
    assert float(result["x"][1]) == 5.0
    assert float(result["fun"]) <= 64.0 + 1e-5


def test_lbfgs_rejects_invalid_initial_state():
    with pytest.raises(FloatingPointError, match="finite"):
        lbfgs_minimize(lambda x: jnp.sum(x), jnp.asarray([jnp.nan]))
    with pytest.raises(ValueError, match="same shape"):
        lbfgs_minimize(
            lambda x: jnp.sum(x * x),
            jnp.ones(2),
            grad_mask=jnp.ones(3),
        )


@dataclass(frozen=True)
class _Pose:
    coords: object
    packed_block_types: object = None

    @property
    def atom_is_real(self):
        return jnp.asarray([[True, True, False]])

    def replace(self, **updates):
        return replace(self, **updates)


class _QuadraticTerm:
    def __init__(self, database):
        pass

    def score_types_provided(self):
        return (ScoreType.fa_ljatr,)

    def setup_for_pose_stack(self, pose, block_types):
        return None

    def __call__(self, pose, state, coords_override=None):
        return {ScoreType.fa_ljatr: (coords_override**2).sum(axis=(1, 2))}


def test_run_cart_min_freezes_mask_and_padding(monkeypatch):
    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QuadraticTerm=_QuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(
        scoring._TERM_FOR_TYPE, ScoreType.fa_ljatr, ("fake", "QuadraticTerm")
    )
    score_function = ScoreFunction(None)
    score_function.set_weight(ScoreType.fa_ljatr, 1.0)
    pose = _Pose(jnp.ones((1, 3, 3)))
    move_mask = jnp.asarray([[True, False, True]])

    minimized, result = run_cart_min(
        pose,
        score_function,
        move_mask,
        max_iter=10,
        return_result=True,
    )
    np.testing.assert_allclose(minimized.coords[:, 0], 0, atol=1e-6)
    np.testing.assert_array_equal(minimized.coords[:, 1:], pose.coords[:, 1:])
    assert result["converged"]
    assert result["initial_fun"] == 9.0
    assert float(result["fun"]) <= 6.0 + 1e-6

    with pytest.raises(ValueError, match="move_mask"):
        run_cart_min(pose, score_function, jnp.ones((3,)), max_iter=0)


def test_real_score_cart_min_is_finite_and_nonincreasing(tmp_path):
    source = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
    ]
    path = tmp_path / "tripeptide.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    pose = pose_stack_from_pdb(path)
    score_function = beta2016_score_function()
    initial = float(score_function.jitted(pose)(pose.coords).sum())

    minimized, result = run_cart_min(
        pose, score_function, max_iter=2, return_result=True
    )
    final = float(score_function.jitted(minimized)(minimized.coords).sum())
    assert np.isfinite(minimized.coords).all()
    assert np.isfinite(final)
    assert final <= initial + 1e-4
    assert float(result["fun"]) <= initial + 1e-4


class _RepulsiveQuadraticTerm(_QuadraticTerm):
    def score_types_provided(self):
        return (ScoreType.fa_ljrep,)

    def __call__(self, pose, state, coords_override=None):
        return {ScoreType.fa_ljrep: (coords_override**2).sum(axis=(1, 2))}


def test_cart_relax_preserves_score_configuration(monkeypatch):
    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(RepulsiveQuadraticTerm=_RepulsiveQuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(
        scoring._TERM_FOR_TYPE,
        ScoreType.fa_ljrep,
        ("fake", "RepulsiveQuadraticTerm"),
    )
    score_function = ScoreFunction(None)
    score_function.set_weight(ScoreType.fa_ljrep, 0.55)
    original_weights = score_function.weights
    pose = _Pose(jnp.ones((1, 3, 3)))

    relaxed, result = run_cart_relax(
        pose,
        score_function,
        ramp_schedule=(0.5, 1.0),
        max_iter=3,
        return_result=True,
    )

    assert score_function.weights == original_weights
    assert len(result["stages"]) == 2
    assert result["fun"] <= result["initial_fun"] + 1e-6
    np.testing.assert_array_equal(relaxed.coords[:, 2], pose.coords[:, 2])


@pytest.mark.parametrize(
    "schedule",
    [(), (0.5,), (0.5, 0.25, 1.0), (0.0, 1.0), (float("nan"), 1.0)],
)
def test_cart_relax_rejects_invalid_ramp(schedule):
    score_function = ScoreFunction(None)
    pose = _Pose(jnp.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="ramp"):
        run_cart_relax(pose, score_function, ramp_schedule=schedule)


def test_explicit_constraint_set_is_rendered():
    pose = _Pose(jnp.ones((1, 3, 3)))
    constraints = ConstraintSet(
        target_coords=jnp.zeros((1, 3)),
        atom_indices=jnp.asarray([[0, 0]], dtype=jnp.int32),
        k=jnp.ones((1,)),
        tol=jnp.zeros((1,)),
    )
    score_function = ScoreFunction(None)
    score_function.set_weight(ScoreType.constraint, 1.0)
    scorer = score_function.render_whole_pose_scoring_module(
        pose, constraint_set=constraints
    )

    np.testing.assert_allclose(scorer(pose.coords), [1.5], atol=1e-6)


def test_real_score_cart_relax_is_finite_and_nonincreasing(tmp_path):
    source = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.startswith("ATOM") and 1 <= int(line[22:26]) <= 3
    ]
    path = tmp_path / "tripeptide-relax.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    pose = pose_stack_from_pdb(path)
    score_function = beta2016_score_function()
    original_weights = score_function.weights

    relaxed, result = run_cart_relax(
        pose,
        score_function,
        constrain_mainchain=True,
        ramp_schedule=(1.0,),
        max_iter=1,
        return_result=True,
    )

    assert score_function.weights == original_weights
    assert np.isfinite(relaxed.coords).all()
    assert np.isfinite(result["fun"])
    assert result["fun"] <= result["initial_fun"] + 1e-4
