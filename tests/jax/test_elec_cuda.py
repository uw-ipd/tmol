from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score import _cuda as cuda_loader
from tmol_jax.score.elec import cuda
from tmol_jax.score.elec.elec_energy_term import ElecEnergyTerm
from tmol_jax.score.score_types import ScoreType


def test_cuda_plugin_is_optional(monkeypatch):
    def missing(_name):
        raise ImportError

    cuda_loader.load_library.cache_clear()
    monkeypatch.setattr(cuda_loader.importlib, "import_module", missing)
    assert cuda_loader.load_library() is None
    cuda_loader.load_library.cache_clear()


def test_pack_cuda_parameters():
    state = SimpleNamespace(
        D=jnp.float32(80),
        D0=jnp.float32(6),
        S=jnp.float32(0.4),
        min_dis=jnp.float32(1.45),
        max_dis=jnp.float32(5.5),
    )
    values = cuda._pack_global_parameters(state)
    assert values.shape == (11,)
    np.testing.assert_array_equal(
        np.asarray(values[:5]), np.asarray([80, 6, 0.4, 1.45, 5.5], dtype=np.float32)
    )
    assert np.isfinite(np.asarray(values)).all()


def test_elec_dispatches_to_installed_cuda_plugin(monkeypatch):
    expected = jnp.asarray([1.0], dtype=jnp.float32)
    seen = {}

    def score(state, coordinates):
        seen["arguments"] = state, coordinates
        return expected

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda, "available", lambda: True)
    monkeypatch.setattr(cuda, "score", score)
    coordinates = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    state = object()
    term = object.__new__(ElecEnergyTerm)
    result = term(SimpleNamespace(coords=coordinates), state)

    assert seen["arguments"][0] is state
    assert seen["arguments"][1] is coordinates
    np.testing.assert_array_equal(result[ScoreType.fa_elec], expected)
