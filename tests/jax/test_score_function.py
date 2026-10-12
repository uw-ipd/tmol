"""Rendered scorers must not silently retain stale weights or scoring state."""

from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_INITIAL_X64 = jax.config.x64_enabled

from tmol_jax.score import score_function as scoring  # noqa: E402
from tmol_jax.score import ScoreFunction, ScoreType  # noqa: E402


class QuadraticTerm:
    def __init__(self, database):
        self.scale = 1.0

    def score_types_provided(self):
        return (ScoreType.fa_ljatr,)

    def setup_for_pose_stack(self, pose, block_types):
        return 1.0

    def __call__(self, pose, state, coords_override=None):
        return {
            ScoreType.fa_ljatr: self.scale
            * state
            * (coords_override**2).sum(axis=(1, 2))
        }


@pytest.fixture
def scorer(monkeypatch):
    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QuadraticTerm=QuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(
        scoring._TERM_FOR_TYPE, ScoreType.fa_ljatr, ("fake", "QuadraticTerm")
    )
    score = ScoreFunction(None)
    score.set_weight(ScoreType.fa_ljatr, 2.0)
    return score


def test_render_freezes_configuration(scorer):
    pose = SimpleNamespace(coords=jnp.ones((2, 3, 3)), packed_block_types=None)
    old = jax.jit(scorer.render_whole_pose_scoring_module(pose))
    scorer.set_weight(ScoreType.fa_ljatr, 3.0)
    scorer.terms[0].scale = 2.0
    new = scorer.jitted(pose)
    # Even first tracing after mutation preserves the original render snapshot.
    np.testing.assert_allclose(old(pose.coords), [18, 18])
    np.testing.assert_allclose(new(pose.coords), [54, 54])
    per_term = scorer.render_whole_pose_scoring_module(
        pose, sum_terms=False, apply_weights=False
    )
    np.testing.assert_allclose(per_term(pose.coords), [[18, 18]])
    energy, grad = scorer.jitted_value_and_grad(pose)(pose.coords)
    np.testing.assert_allclose(energy, [54, 54])
    np.testing.assert_allclose(grad, np.full((2, 3, 3), 12.0))


def test_zero_weight_and_shape_validation(scorer):
    pose = SimpleNamespace(coords=jnp.ones((1, 2, 3)), packed_block_types=None)
    scorer.set_weight(ScoreType.fa_ljatr, 0)
    fn = scorer.jitted(pose)
    np.testing.assert_array_equal(fn(pose.coords), [0])
    np.testing.assert_array_equal(jax.grad(lambda c: fn(c).sum())(pose.coords), 0)
    with pytest.raises(ValueError, match="shape"):
        fn(jnp.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="finite"):
        scorer.set_weight(ScoreType.fa_ljatr, float("nan"))


def test_import_does_not_load_torch_or_enable_global_x64():
    assert "torch" not in sys.modules
    assert jax.config.x64_enabled == _INITIAL_X64


def test_setup_cache_checks_owners_and_releases_dead_poses():
    import gc
    from collections import OrderedDict
    from tmol_jax.score.common.count_pair import _cache_get, _cache_put

    class Owner:
        pass

    cache = OrderedDict()
    owner = Owner()
    replacement = Owner()
    _cache_put(cache, 7, "old", (owner,))
    assert _cache_get(cache, 7, (replacement,)) is None
    _cache_put(cache, 7, "new", (replacement,))
    del replacement
    gc.collect()
    assert not cache


def test_count_pair_storage_is_compact_and_large_clamps_remain_lossless():
    from tmol_jax.score.common.count_pair import _pairwise_bpl_jit

    arguments = (
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.full((1, 1, 1, 1, 1), 6, dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([[[0, 1], [1, 0]]], dtype=jnp.int32),
        2,
    )

    canonical = _pairwise_bpl_jit(*arguments, max_sep=6)
    assert canonical.dtype == jnp.uint8
    np.testing.assert_array_equal(
        canonical, jnp.asarray([[[0, 1], [1, 0]]], dtype=jnp.uint8)
    )

    oversized = _pairwise_bpl_jit(*arguments, max_sep=256)
    assert oversized.dtype == jnp.int32
    np.testing.assert_array_equal(
        oversized, jnp.asarray([[[0, 1], [1, 0]]], dtype=jnp.int32)
    )
