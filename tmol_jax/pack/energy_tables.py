"""Exact assignment energies for TMol packer interaction tables."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tmol_jax.types import Array, pytree_dataclass


@pytree_dataclass
class PackerEnergyTables:
    """Canonical sparse one-/two-body tables consumed by TMol's annealer."""

    nrotamers_for_res: Array
    oneb_offsets: Array
    chunk_size: int
    neighbor_row_offsets: Array
    neighbor_blocks: Array
    neighbor_chunk_offset_offsets: Array
    chunk_offsets: Array
    energy1b: Array
    energy2b: Array


def assignment_energies(tables: PackerEnergyTables, assignments: Array) -> jax.Array:
    """Return exact table energy for local rotamer assignments.

    ``assignments`` has shape ``[pose, trajectory, residue]``. Entries are local
    rotamer indices; ``-1`` denotes a fixed, absent, or padded residue. Table
    indexing matches ``tmol::pack::compiled::total_energy_for_assignment``.
    Invalid local indices produce ``NaN`` instead of silently omitting a residue.
    """
    assignments = jnp.asarray(assignments, dtype=jnp.int32)
    nrotamers = jnp.asarray(tables.nrotamers_for_res, dtype=jnp.int32)
    oneb_offsets = jnp.asarray(tables.oneb_offsets, dtype=jnp.int32)
    if assignments.ndim != 3 or nrotamers.ndim != 2:
        raise ValueError("assignments must be rank 3 and nrotamers_for_res rank 2")
    n_poses, n_trajectories, max_n_res = assignments.shape
    if nrotamers.shape != (n_poses, max_n_res):
        raise ValueError("assignment and energy-table pose/residue shapes differ")
    if oneb_offsets.shape != nrotamers.shape:
        raise ValueError("oneb_offsets must match nrotamers_for_res")
    if tables.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    invalid = (assignments < -1) | (
        (assignments >= nrotamers[:, None, :]) & (assignments != -1)
    )
    active = (assignments >= 0) & ~invalid
    local = jnp.where(active, assignments, 0)
    energy1b = jnp.asarray(tables.energy1b)
    if energy1b.shape[0] == 0:
        oneb = jnp.zeros((n_poses, n_trajectories), dtype=jnp.float32)
    else:
        oneb_ind = oneb_offsets[:, None, :] + local
        oneb_value = energy1b[jnp.clip(oneb_ind, 0, energy1b.size - 1)]
        oneb = jnp.where(active, oneb_value, 0).sum(axis=-1)

    n_edges = tables.neighbor_blocks.shape[0]
    if n_edges == 0 or tables.energy2b.shape[0] == 0:
        return jnp.where(invalid.any(axis=-1), jnp.nan, oneb)

    row_offsets = jnp.asarray(tables.neighbor_row_offsets)
    n_rows = n_poses * max_n_res
    if row_offsets.shape != (n_rows + 1,):
        raise ValueError("neighbor_row_offsets has the wrong number of CSR rows")
    row = jnp.repeat(
        jnp.arange(n_rows, dtype=jnp.int32),
        jnp.diff(row_offsets),
        total_repeat_length=n_edges,
    )
    edge_pose = row // max_n_res
    edge_i = row % max_n_res
    edge_j = jnp.asarray(tables.neighbor_blocks, dtype=jnp.int32)

    irot = local[edge_pose, :, edge_i]
    jrot = local[edge_pose, :, edge_j]
    valid = (
        (edge_j > edge_i)[:, None]
        & active[edge_pose, :, edge_i]
        & active[edge_pose, :, edge_j]
    )

    chunk_size = tables.chunk_size
    i_chunk, i_in_chunk = divmod(irot, chunk_size)
    j_chunk, j_in_chunk = divmod(jrot, chunk_size)
    j_nrot = nrotamers[edge_pose, edge_j][:, None]
    j_nchunks = jnp.maximum(1, (j_nrot - 1) // chunk_size + 1)
    chunk_table_ind = (
        jnp.asarray(tables.neighbor_chunk_offset_offsets)[:, None]
        + i_chunk * j_nchunks
        + j_chunk
    )
    chunk_offsets = jnp.asarray(tables.chunk_offsets)
    chunk_offset = chunk_offsets[jnp.clip(chunk_table_ind, 0, chunk_offsets.size - 1)]
    valid &= chunk_offset >= 0
    j_chunk_size = jnp.minimum(chunk_size, j_nrot - j_chunk * chunk_size)
    pair_ind = chunk_offset + i_in_chunk * j_chunk_size + j_in_chunk
    energy2b = jnp.asarray(tables.energy2b)
    pair = jnp.where(valid, energy2b[jnp.clip(pair_ind, 0, energy2b.size - 1)], 0)
    pair_by_pose = jnp.zeros((n_poses, n_trajectories), dtype=energy2b.dtype)
    pair_by_pose = pair_by_pose.at[edge_pose].add(pair)
    total = oneb + pair_by_pose
    return jnp.where(invalid.any(axis=-1), jnp.nan, total)
