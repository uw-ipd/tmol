from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.score.elec import cuda as elec_cuda
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


@pytest.mark.skipif(
    jax.default_backend() != "gpu"
    or not cuda.fused_available()
    or not elec_cuda.available(),
    reason="optional CUDA fused pair kernel is unavailable",
)
def test_fused_tile_bounds_recompute_after_large_coordinate_move():
    """Tile pruning is rebuilt from every coordinate argument, not frozen."""
    n_atoms = 2048
    values = jnp.ones((1, n_atoms), dtype=jnp.float32)
    real = jnp.zeros((1, n_atoms), dtype=jnp.bool_).at[0, (0, n_atoms - 1)].set(True)
    paths = jnp.full((1, n_atoms, n_atoms), 6, dtype=jnp.uint8)
    parameters = SimpleNamespace(
        lj_radius=values * 1.5,
        lj_wdepth=values,
        lk_dgfree=values * -0.1,
        lk_lambda=values * 3.5,
        lk_volume=values * 12.0,
        is_donor=values < 0,
        is_hydroxyl=values < 0,
        is_polarh=values < 0,
        is_acceptor=values < 0,
        is_carbon_lk=values < 0,
    )
    ljlk_state = SimpleNamespace(
        parameters=parameters,
        global_parameters=SimpleNamespace(
            max_dis=jnp.float32(6),
            lj_dlin_sigma_factor=jnp.float32(0.6),
            lj_hbond_dis=jnp.float32(3),
            lj_hbond_OH_donor_dis=jnp.float32(2.8),
            lj_hbond_hdis=jnp.float32(1.9),
        ),
        bonded_path_length=paths,
        atom_is_real=real,
    )
    elec_parameters = SimpleNamespace(
        D=jnp.float32(80),
        D0=jnp.float32(6),
        S=jnp.float32(0.4),
        min_dis=jnp.float32(1.5),
        max_dis=jnp.float32(5.5),
    )
    elec_state = SimpleNamespace(
        charges=values.at[0, n_atoms - 1].set(-1),
        bonded_path_length=paths,
        atom_is_real=real,
        cuda_global_parameters=elec_cuda._pack_global_parameters(elec_parameters),
    )
    far = jnp.zeros((1, n_atoms, 3), dtype=jnp.float32)
    far = far.at[0, n_atoms - 1, 0].set(20)
    near = far.at[0, n_atoms - 1, 0].set(4)
    weights = jnp.asarray([0.7, -0.2, 1.3, 0.5], dtype=jnp.float32)

    def separate(coordinates):
        ljlk = cuda.score(ljlk_state, coordinates)
        elec = elec_cuda.score(elec_state, coordinates)
        return jnp.concatenate((ljlk, elec[:, None]), axis=1)

    def fused(coordinates):
        return cuda.score_fused(ljlk_state, elec_state, coordinates)

    separate_vg = jax.jit(jax.value_and_grad(lambda x: (separate(x) * weights).sum()))
    fused_vg = jax.jit(jax.value_and_grad(lambda x: (fused(x) * weights).sum()))
    for coordinates in (far, near):
        separate_value, separate_gradient = separate_vg(coordinates)
        fused_value, fused_gradient = fused_vg(coordinates)
        np.testing.assert_allclose(fused_value, separate_value, atol=1e-5, rtol=1e-6)
        np.testing.assert_allclose(
            fused_gradient, separate_gradient, atol=1e-5, rtol=1e-5
        )
    assert fused_vg(far)[0] == 0
    assert fused_vg(near)[0] != 0
