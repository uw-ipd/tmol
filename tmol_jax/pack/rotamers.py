"""Exact coordinate imposition for position-specific repacking rotamers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tmol_jax.pack.energy_tables import run_metropolis_trace
from tmol_jax.types import Array, pytree_dataclass


@pytree_dataclass
class RepackingRotamers:
    """Coordinates and ownership metadata for same-type rotamer candidates."""

    pose_for_rot: Array
    block_ind_for_rot: Array
    block_type_ind_for_rot: Array
    coord_offset_for_rot: Array
    coords: Array


def impose_repacking_assignments(pose_stack, rotamers, assignments):
    """Return a pose with one position-specific rotamer imposed per block.

    ``assignments`` contains global indices into ``rotamers`` and has shape
    ``[pose, block]``; padding blocks must contain ``-1``. This deliberately
    supports repacking only. Candidates from another pose/block or with a
    different residue type are rejected rather than silently used as design.
    """
    assignment_np = np.asarray(assignments, dtype=np.int32)
    block_type_np = np.asarray(pose_stack.block_type_ind, dtype=np.int32)
    expected_shape = tuple(block_type_np.shape)
    if assignment_np.shape != expected_shape:
        raise ValueError(
            f"assignments shape {assignment_np.shape} does not match {expected_shape}"
        )

    real = block_type_np >= 0
    if np.any(assignment_np[~real] != -1):
        raise ValueError("padding blocks must have assignment -1")
    n_rotamers = int(np.asarray(rotamers.block_type_ind_for_rot).shape[0])
    if np.any(assignment_np[real] < 0) or np.any(assignment_np[real] >= n_rotamers):
        raise ValueError("every real block must have a valid rotamer assignment")

    chosen = assignment_np[real]
    chosen_pose = np.asarray(rotamers.pose_for_rot, dtype=np.int32)[chosen]
    chosen_block = np.asarray(rotamers.block_ind_for_rot, dtype=np.int32)[chosen]
    chosen_type = np.asarray(rotamers.block_type_ind_for_rot, dtype=np.int32)[chosen]
    expected_pose, expected_block = np.nonzero(real)
    if np.any(chosen_pose != expected_pose) or np.any(chosen_block != expected_block):
        raise ValueError("a rotamer assignment belongs to a different pose/block")
    if np.any(chosen_type != block_type_np[real]):
        raise ValueError("residue-type changes are not supported by repacking")

    assignment = jnp.asarray(assignments, dtype=jnp.int32)
    safe_assignment = jnp.maximum(assignment, 0)
    block_type = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    safe_type = jnp.maximum(block_type, 0)
    n_atoms = jnp.asarray(pose_stack.packed_block_types.n_atoms, dtype=jnp.int32)[
        safe_type
    ]
    n_atoms = jnp.where(block_type >= 0, n_atoms, 0)
    atom = jnp.arange(pose_stack.max_n_atoms_per_block, dtype=jnp.int32)
    is_real_atom = atom[None, None, :] < n_atoms[..., None]

    source = (
        jnp.asarray(rotamers.coord_offset_for_rot, dtype=jnp.int32)[safe_assignment][
            ..., None
        ]
        + atom
    )
    source = jnp.where(is_real_atom, source, 0)
    destination = (
        jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)[..., None] + atom
    )
    destination = jnp.where(is_real_atom, destination, pose_stack.max_n_atoms_per_pose)
    pose = jnp.broadcast_to(
        jnp.arange(pose_stack.n_poses, dtype=jnp.int32)[:, None, None],
        destination.shape,
    )
    coords = jnp.zeros_like(pose_stack.coords)
    coords = coords.at[pose, destination].set(
        jnp.asarray(rotamers.coords)[source], mode="drop"
    )
    return pose_stack.replace(coords=coords)


def repack_from_trace(
    pose_stack,
    rotamers,
    energy_tables,
    initial_assignments,
    proposal_residues,
    proposal_rotamers,
    temperatures,
    uniform_randoms,
    quenches,
    *,
    base_candidate_assignments,
    candidate_for_table_rotamer,
    block_for_table_residue,
):
    """Anneal and impose the best supplied same-type rotamer assignments.

    Assignments and proposals use position-local rotamer indices. The proposal
    trace is explicit so callers can reproduce the same stochastic decisions
    in another frontend. Candidate generation, energy-table construction, and
    random schedule generation deliberately remain outside this function.
    The explicit mappings preserve canonical bump-checked rotamer numbering,
    molten-block numbering, and fixed-block assignments.
    """
    _, best, best_energies, accepted = run_metropolis_trace(
        energy_tables,
        initial_assignments,
        proposal_residues,
        proposal_rotamers,
        temperatures,
        uniform_randoms,
        quenches,
    )
    best_trajectory = jnp.argmin(best_energies, axis=1)
    best_local = jnp.take_along_axis(best, best_trajectory[:, None, None], axis=1)[:, 0]
    best_energy = jnp.take_along_axis(best_energies, best_trajectory[:, None], axis=1)[
        :, 0
    ]
    table_rotamer = jnp.where(
        best_local >= 0,
        best_local + jnp.asarray(energy_tables.oneb_offsets, dtype=jnp.int32),
        -1,
    )
    candidate_for_table_rotamer = np.asarray(
        candidate_for_table_rotamer, dtype=np.int32
    )
    table_rotamer_np = np.asarray(table_rotamer, dtype=np.int32)
    active = table_rotamer_np >= 0
    if np.any(table_rotamer_np[active] >= candidate_for_table_rotamer.shape[0]):
        raise ValueError("table rotamer mapping is incomplete")
    chosen_candidate = np.full(table_rotamer_np.shape, -1, dtype=np.int32)
    chosen_candidate[active] = candidate_for_table_rotamer[table_rotamer_np[active]]

    block = np.asarray(block_for_table_residue, dtype=np.int32)
    if block.shape != table_rotamer_np.shape:
        raise ValueError("table residue mapping has the wrong shape")
    assignments = np.asarray(base_candidate_assignments, dtype=np.int32).copy()
    pose = np.broadcast_to(np.arange(assignments.shape[0])[:, None], block.shape)
    molten = block >= 0
    assignments[pose[molten], block[molten]] = chosen_candidate[molten]
    packed = impose_repacking_assignments(pose_stack, rotamers, assignments)
    return packed, best_local, best_energy, accepted
