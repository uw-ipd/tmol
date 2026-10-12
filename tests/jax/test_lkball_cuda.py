from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.lk_ball import cuda
from tmol_jax.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol_jax.score.score_types import ScoreType


def test_cuda_plugin_is_optional(monkeypatch):
    def missing(_name):
        raise ImportError

    cuda._register_library.cache_clear()
    monkeypatch.setattr(cuda.importlib, "import_module", missing)
    assert not cuda.available()
    cuda._register_library.cache_clear()


def test_pack_cuda_parameters():
    values = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
    state = SimpleNamespace(
        lj_radius=values,
        lk_dgfree=values,
        lk_lambda=values,
        lk_volume=values,
        is_donor=values,
        is_hydroxyl=values,
        is_polarh=values,
        is_acceptor=values,
        is_carbon_lk=values,
    )
    defaults = SimpleNamespace(
        lj_hbond_dis=3.0,
        lj_hbond_OH_donor_dis=2.8,
        lj_hbond_hdis=1.9,
        lkb_water_dist=2.65,
        distance_threshold=6.0,
    )
    packed, global_values = cuda._pack_parameters(
        SimpleNamespace(defaults=defaults), state
    )
    assert packed.shape == (1, 2, 9)
    np.testing.assert_array_equal(np.asarray(packed[..., 0]), np.asarray(values))
    np.testing.assert_array_equal(np.asarray(packed[..., 8]), np.asarray(values))
    np.testing.assert_allclose(
        np.asarray(global_values), [3, 2.8, 1.9, 2.65, 6], rtol=0, atol=1e-7
    )


def test_lkball_dispatches_to_installed_cuda_plugin(monkeypatch):
    expected = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    seen = {}

    def score(term, state, coordinates):
        seen["arguments"] = term, state, coordinates
        return expected

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda, "available", lambda: True)
    monkeypatch.setattr(cuda, "score", score)
    coordinates = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    state = object()
    term = object.__new__(LKBallEnergyTerm)
    result = term._eval_full(coordinates, state)

    assert seen["arguments"][0] is term
    assert seen["arguments"][1] is state
    assert seen["arguments"][2] is coordinates
    for index, score_type in enumerate(
        (
            ScoreType.lk_ball_iso,
            ScoreType.lk_ball,
            ScoreType.lk_bridge,
            ScoreType.lk_bridge_uncpl,
        )
    ):
        np.testing.assert_array_equal(result[score_type], expected[:, index])
