from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score import _cuda as cuda_loader
from tmol_jax.score.hbond import cuda
from tmol_jax.score.hbond.hbond_energy_term import HBondEnergyTerm
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
        atom_params=SimpleNamespace(
            donor_H_atom=jnp.asarray([[1]], dtype=jnp.int32),
            donor_heavy_atom=jnp.asarray([[2]], dtype=jnp.int32),
            donor_type=jnp.asarray([[3]], dtype=jnp.int32),
            acceptor_atom=jnp.asarray([[4]], dtype=jnp.int32),
            acceptor_base=jnp.asarray([[5]], dtype=jnp.int32),
            acceptor_base0=jnp.asarray([[6]], dtype=jnp.int32),
            acceptor_type=jnp.asarray([[7]], dtype=jnp.int32),
        ),
        pair_params=SimpleNamespace(
            weight=jnp.asarray([[0.5]], dtype=jnp.float32),
            acceptor_hybridization=jnp.asarray([[2]], dtype=jnp.int32),
        ),
        global_params=SimpleNamespace(
            hb_sp2_range_span=jnp.float32(1),
            hb_sp2_BAH180_rise=jnp.float32(2),
            hb_sp2_outer_width=jnp.float32(3),
            hb_sp3_softmax_fade=jnp.float32(4),
            threshold_distance=jnp.float32(5),
            max_ah_dist=jnp.float32(6),
        ),
    )
    donors, acceptors, pairs, globals_ = cuda._pack_parameters(state)
    np.testing.assert_array_equal(np.asarray(donors), [[[1, 2, 3]]])
    np.testing.assert_array_equal(np.asarray(acceptors), [[[4, 5, 6, 7]]])
    np.testing.assert_array_equal(np.asarray(pairs), [[[0.5, 2]]])
    np.testing.assert_array_equal(np.asarray(globals_), [1, 2, 3, 4, 5, 6])


def test_hbond_dispatches_to_installed_cuda_plugin(monkeypatch):
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
    term = object.__new__(HBondEnergyTerm)
    result = term(SimpleNamespace(coords=coordinates), state)

    assert seen["arguments"][0] is state
    assert seen["arguments"][1] is coordinates
    np.testing.assert_array_equal(result[ScoreType.hbond], expected)
