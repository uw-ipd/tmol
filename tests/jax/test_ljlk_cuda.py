from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.ljlk import cuda
from tmol_jax.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
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
    parameters = SimpleNamespace(
        lj_radius=values,
        lj_wdepth=values * values,
        lk_dgfree=values,
        lk_lambda=values,
        lk_volume=values,
        is_donor=values > 0,
        is_hydroxyl=values > 0,
        is_polarh=values > 0,
        is_acceptor=values > 0,
        is_carbon_lk=values > 0,
    )
    global_parameters = SimpleNamespace(
        max_dis=jnp.float32(6),
        lj_dlin_sigma_factor=jnp.float32(0.6),
        lj_hbond_dis=jnp.float32(3),
        lj_hbond_OH_donor_dis=jnp.float32(2.8),
        lj_hbond_hdis=jnp.float32(1.9),
    )
    packed, global_values = cuda._pack_parameters(
        SimpleNamespace(parameters=parameters, global_parameters=global_parameters)
    )
    assert packed.shape == (1, 2, 13)
    np.testing.assert_array_equal(np.asarray(packed[..., 0]), np.asarray(values))
    np.testing.assert_array_equal(np.asarray(packed[..., 1]), np.asarray(values))
    np.testing.assert_array_equal(np.asarray(packed[..., 10]), 0)
    np.testing.assert_allclose(
        np.asarray(global_values), [6, 0.6, 3, 2.8, 1.9], rtol=0, atol=1e-7
    )


def test_ljlk_dispatches_to_installed_cuda_plugin(monkeypatch):
    expected = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    seen = {}

    def score(state, coordinates):
        seen["arguments"] = state, coordinates
        return expected

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda, "available", lambda: True)
    monkeypatch.setattr(cuda, "score", score)
    coordinates = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    state = object()
    term = object.__new__(LJLKEnergyTerm)
    result = term(SimpleNamespace(coords=coordinates), state)

    assert seen["arguments"][0] is state
    assert seen["arguments"][1] is coordinates
    np.testing.assert_array_equal(result[ScoreType.fa_ljatr], expected[:, 0])
    np.testing.assert_array_equal(result[ScoreType.fa_ljrep], expected[:, 1])
    np.testing.assert_array_equal(result[ScoreType.fa_lk], expected[:, 2])
