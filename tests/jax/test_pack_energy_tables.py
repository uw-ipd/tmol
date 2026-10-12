import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.pack import PackerEnergyTables, assignment_energies, run_metropolis_trace


def _mixed_tables():
    return PackerEnergyTables(
        nrotamers_for_res=jnp.asarray([[3, 2, 0], [1, 0, 2]], dtype=jnp.int32),
        oneb_offsets=jnp.asarray([[0, 3, 0], [5, 0, 6]], dtype=jnp.int32),
        chunk_size=2,
        neighbor_row_offsets=jnp.asarray([0, 1, 2, 2, 3, 3, 4], dtype=jnp.int32),
        neighbor_blocks=jnp.asarray([1, 0, 2, 0], dtype=jnp.int32),
        neighbor_chunk_offset_offsets=jnp.asarray([0, 2, 4, 5], dtype=jnp.int32),
        chunk_offsets=jnp.asarray([0, 4, 0, 0, 6, 6], dtype=jnp.int32),
        energy1b=jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.float32),
        energy2b=jnp.asarray([10, 11, 12, 13, 14, 15, 20, 21], dtype=jnp.float32),
    )


def test_assignment_energies_mixed_fixed_and_padded():
    assignments = jnp.asarray(
        [
            [[0, 0, -1], [2, 1, -1], [1, -1, -1]],
            [[0, -1, 0], [0, -1, 1], [-1, -1, 1]],
        ],
        dtype=jnp.int32,
    )
    expected = np.asarray([[15, 23, 2], [33, 35, 8]], dtype=np.float32)
    observed = jax.jit(assignment_energies)(_mixed_tables(), assignments)
    np.testing.assert_array_equal(observed, expected)


def test_assignment_energies_without_pair_edges():
    tables = PackerEnergyTables(
        nrotamers_for_res=jnp.asarray([[2, 1]], dtype=jnp.int32),
        oneb_offsets=jnp.asarray([[0, 2]], dtype=jnp.int32),
        chunk_size=2,
        neighbor_row_offsets=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        neighbor_blocks=jnp.empty((0,), dtype=jnp.int32),
        neighbor_chunk_offset_offsets=jnp.empty((0,), dtype=jnp.int32),
        chunk_offsets=jnp.empty((0,), dtype=jnp.int32),
        energy1b=jnp.asarray([1, 2, 3], dtype=jnp.float32),
        energy2b=jnp.empty((0,), dtype=jnp.float32),
    )
    assignments = jnp.asarray([[[0, 0], [1, -1]]], dtype=jnp.int32)
    np.testing.assert_array_equal(assignment_energies(tables, assignments), [[4, 2]])


def test_assignment_energies_reject_invalid_local_index():
    assignments = jnp.asarray([[[3, 0, -1]], [[0, -2, 0]]], dtype=jnp.int32)
    assert np.isnan(np.asarray(assignment_energies(_mixed_tables(), assignments))).all()


def test_assignment_energies_all_fixed_empty_tables():
    tables = PackerEnergyTables(
        nrotamers_for_res=jnp.asarray([[0]], dtype=jnp.int32),
        oneb_offsets=jnp.asarray([[0]], dtype=jnp.int32),
        chunk_size=1,
        neighbor_row_offsets=jnp.asarray([0, 0], dtype=jnp.int32),
        neighbor_blocks=jnp.empty((0,), dtype=jnp.int32),
        neighbor_chunk_offset_offsets=jnp.empty((0,), dtype=jnp.int32),
        chunk_offsets=jnp.empty((0,), dtype=jnp.int32),
        energy1b=jnp.empty((0,), dtype=jnp.float32),
        energy2b=jnp.empty((0,), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(assignment_energies(tables, [[[-1]]]), [[0]])


def test_run_metropolis_trace_matches_canonical_acceptance_rule():
    initial = jnp.asarray(
        [[[0, 0, -1]], [[0, -1, 0]]],
        dtype=jnp.int32,
    )
    proposal_residues = jnp.asarray(
        [[[0], [2]], [[0], [2]], [[0], [2]]], dtype=jnp.int32
    )
    proposal_rotamers = jnp.asarray(
        [[[1], [1]], [[1], [0]], [[1], [0]]], dtype=jnp.int32
    )
    temperatures = jnp.ones((3, 2, 1), dtype=jnp.float32)
    uniforms = jnp.asarray(
        [[[0.8], [0.9]], [[0.0], [0.99]], [[0.7], [0.5]]],
        dtype=jnp.float32,
    )
    quenches = jnp.asarray([[[False], [False]], [[True], [True]], [[False], [False]]])

    final, best, best_energy, accepted = jax.jit(run_metropolis_trace)(
        _mixed_tables(),
        initial,
        proposal_residues,
        proposal_rotamers,
        temperatures,
        uniforms,
        quenches,
    )

    np.testing.assert_array_equal(
        accepted[..., 0],
        [[False, True], [False, True], [True, True]],
    )
    np.testing.assert_array_equal(
        final,
        [[[1, 0, -1]], [[0, -1, 0]]],
    )
    np.testing.assert_array_equal(best, initial)
    np.testing.assert_array_equal(best_energy, [[15], [33]])
