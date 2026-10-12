"""Exact coordinate imposition for position-specific repacking rotamers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
