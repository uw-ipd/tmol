"""Constraint-building helpers.

Upstream reference: :mod:`tmol.score.constraint.utility`. The torch
version offers two helpers:

* ``constrain_all_ca`` -- tether every CA to its current coordinate.
* ``create_mainchain_coordinate_constraints`` -- tether every mainchain
  atom (N, CA, C for canonical residues) to its current coordinate. This
  is what ``FastRelax`` uses when it wants to freeze the backbone and
  only repack / minimise sidechains.

This port focuses on the FastRelax use case: given a
:class:`~tmol_jax.pose.pose_stack.PoseStack`, walk the palette to find
each residue's mainchain atom indices, gather the current coordinates,
and pack everything into a :class:`ConstraintSet`.

Because the JAX port of :class:`PackedBlockTypes` does not precompute a
``mainchain_atom_indices`` attribute (the upstream field), we recover
the indices from the per-block-type ``atom_names`` tuple. The default
mainchain atoms for the canonical amino acids are ``N``, ``CA``, ``C``;
an optional ``atom_names`` argument lets callers pick a different set
(e.g. include ``O`` or restrict to ``CA`` only). This keeps the helper
self-contained -- no mutation of the packed block types, no hidden
side-effects.

All heavy lifting runs once on CPU / numpy (the lookup is
conformation-independent and happens outside the jit boundary). Only
the final coordinate gather sits on a JAX array, which keeps the
resulting :class:`ConstraintSet` a proper pytree.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from tmol_jax.score.constraint.constraint_energy_term import ConstraintSet

# Default mainchain atom names for the canonical 20 amino acids.
# Matches the ``properties.polymer.mainchain_atoms`` set used upstream.
_DEFAULT_MAINCHAIN_ATOMS: tuple[str, ...] = ("N", "CA", "C")


def _mainchain_indices_per_block_type(
    pbt: Any,
    atom_names: Sequence[str],
) -> list[list[int]]:
    """Return, for each block type, the local atom indices of the
    requested mainchain atoms (in the order given by ``atom_names``).

    Any requested atom that does not exist on a given block type is
    silently skipped -- non-polymeric residues with no backbone will
    simply contribute zero constraints. The returned lists are Python
    objects (one list per block type) because per-block-type sizes vary
    and the downstream caller flattens them anyway.
    """
    per_type: list[list[int]] = []
    for names in pbt.atom_names:
        # ``names`` is a tuple padded to ``max_n_atoms`` with "" -- empty
        # strings cannot match a real atom name so the ``index`` lookup
        # below is safe.
        wanted: list[int] = []
        for nm in atom_names:
            try:
                wanted.append(names.index(nm))
            except ValueError:
                # This block type lacks this mainchain atom (e.g. a
                # non-polymeric ligand); skip it.
                continue
        per_type.append(wanted)
    return per_type


def create_mainchain_coordinate_constraints(
    ps: Any,
    k: float = 1.0,
    tol: float = 0.0,
    atom_names: Sequence[str] = _DEFAULT_MAINCHAIN_ATOMS,
) -> ConstraintSet:
    """Build a :class:`ConstraintSet` tethering all mainchain atoms
    back to their current positions.

    This is the FastRelax "freeze backbone" restraint: every N, CA, and
    C atom in every real block of every pose gets an individual harmonic
    coordinate constraint pointing at its current coordinate.

    Parameters
    ----------
    ps
        :class:`~tmol_jax.pose.pose_stack.PoseStack`. Read-only; the
        returned :class:`ConstraintSet` closes over a *copy* of the
        relevant coordinates.
    k
        Spring constant ``k`` for every constraint. Default ``1.0`` --
        FastRelax tunes this across its ramps.
    tol
        Flat-bottom tolerance (Angstroms). Default ``0.0`` gives a pure
        harmonic. Nonzero values let the backbone wiggle freely within
        ``tol`` Angstroms of its starting point.
    atom_names
        Names of the mainchain atoms to restrain. Defaults to
        ``("N", "CA", "C")`` to match upstream FastRelax.

    Returns
    -------
    ConstraintSet
        Pytree ready to be passed to
        :meth:`ConstraintEnergyTerm.setup_for_pose_stack`.
    """
    pbt = ps.packed_block_types

    # Per-block-type local atom indices for each requested mainchain atom.
    mc_per_type = _mainchain_indices_per_block_type(pbt, atom_names)

    # Move the arrays we need onto numpy for the (static) bookkeeping
    # loop. These are small and conformation-independent, so numpy is
    # cheaper than repeated ``jnp`` gathers.
    block_type_ind = np.asarray(ps.block_type_ind)  # int32[P, B]
    block_coord_offset = np.asarray(ps.block_coord_offset)  # int32[P, B]
    coords_np = np.asarray(ps.coords)  # float32[P, N, 3]

    n_poses, max_n_blocks = block_type_ind.shape

    pose_inds: list[int] = []
    atom_inds: list[int] = []  # flat per-pose atom index
    target_xyz: list[np.ndarray] = []

    for p in range(n_poses):
        for b in range(max_n_blocks):
            bt = int(block_type_ind[p, b])
            if bt < 0:
                continue  # padding / sentinel block
            off = int(block_coord_offset[p, b])
            for local_atom in mc_per_type[bt]:
                flat = off + local_atom
                pose_inds.append(p)
                atom_inds.append(flat)
                target_xyz.append(coords_np[p, flat])

    n_constraints = len(pose_inds)

    if n_constraints == 0:
        return ConstraintSet.empty()

    atom_indices_np = np.stack(
        [np.asarray(pose_inds, dtype=np.int32), np.asarray(atom_inds, dtype=np.int32)],
        axis=1,
    )  # [n_constraints, 2]

    target_np = np.stack(target_xyz, axis=0).astype(np.float32)
    k_np = np.full((n_constraints,), float(k), dtype=np.float32)
    tol_np = np.full((n_constraints,), float(tol), dtype=np.float32)

    return ConstraintSet(
        target_coords=jnp.asarray(target_np),
        atom_indices=jnp.asarray(atom_indices_np),
        k=jnp.asarray(k_np),
        tol=jnp.asarray(tol_np),
    )


__all__ = ["create_mainchain_coordinate_constraints"]
