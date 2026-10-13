"""Constraint energy term (pure JAX).

Upstream reference: :mod:`tmol.score.constraint.constraint_energy_term`.
The torch version dispatches between several constraint functional forms
(``harmonic``, ``harmonic_coordinate``, ``bounded``, ``circularharmonic``)
and tracks rotamer-aware atom bookkeeping (block indices, per-block
rotamer expansion, sparse per-block-pair output).

For the JAX port we take the simplest path that covers the use case
:func:`FastRelax` cares about: **coordinate constraints** that tether
specific (pose, atom) locations back to target 3-D coordinates with a
linear-above-tolerance harmonic penalty.

A single :class:`ConstraintSet` pytree carries everything the term
needs:

* ``target_coords``  ``float32[n_constraints, 3]``
* ``atom_indices``   ``int32[n_constraints, 2]`` -- ``(pose_ind, atom_ind)``
* ``k``              ``float32[n_constraints]``  -- spring constant
* ``tol``            ``float32[n_constraints]``  -- flat-bottom tolerance

The penalty for constraint ``i`` is::

    penalty_i = 0.5 * k_i * max(0, |coords[pose_i, atom_i] - target_i| - tol_i) ** 2

Energies are summed per-pose and returned under
``ScoreType.constraint``. The term is jit / grad safe: no Python
branching on traced values, all gathers use standard ``jnp`` indexing,
and the empty-constraint case (``n_constraints == 0``) produces a zero
vector of the right shape automatically.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

# --------------------------------------------------------------------- #
# ConstraintSet pytree                                                  #
# --------------------------------------------------------------------- #


@pytree_dataclass
class ConstraintSet:
    """Flat container for a bag of coordinate constraints.

    All fields are ``jax.Array`` leaves, so the whole object flows
    through ``jit`` / ``vmap`` / ``grad`` as a pytree. The leading
    ``n_constraints`` dimension may be zero; every reduction in the
    scoring path returns zeros in that case.

    Fields
    ------
    target_coords : float32[n_constraints, 3]
        Target 3-D coordinate each atom is tethered to.
    atom_indices : int32[n_constraints, 2]
        ``(pose_ind, atom_ind)`` pair identifying the constrained atom
        inside ``PoseStack.coords``. ``atom_ind`` is the flat index into
        a pose's coord slab (i.e. ``block_coord_offset[pose, block] +
        local_atom_ind``).
    k : float32[n_constraints]
        Spring constant for the harmonic penalty.
    tol : float32[n_constraints]
        Flat-bottom tolerance. Distances below this incur no penalty.
    """

    target_coords: jax.Array
    atom_indices: jax.Array
    k: jax.Array
    tol: jax.Array

    # ------------------------------------------------------------- #
    # Convenience constructors                                      #
    # ------------------------------------------------------------- #

    @classmethod
    def empty(cls) -> "ConstraintSet":
        """An empty constraint set (no penalties, any pose stack)."""
        return cls(
            target_coords=jnp.zeros((0, 3), dtype=jnp.float32),
            atom_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            k=jnp.zeros((0,), dtype=jnp.float32),
            tol=jnp.zeros((0,), dtype=jnp.float32),
        )

    @property
    def n_constraints(self) -> int:
        return int(self.atom_indices.shape[0])


# --------------------------------------------------------------------- #
# ConstraintEnergyTerm                                                  #
# --------------------------------------------------------------------- #


class ConstraintEnergyTerm(EnergyTerm):
    """Sum-of-coordinate-restraints energy term.

    Parameters
    ----------
    param_db
        Ignored (retained for :class:`EnergyTerm` compatibility). The
        constraint set is user-supplied; there are no tabulated
        parameters to look up.
    """

    def __init__(self, param_db: Any = None):
        super().__init__(param_db=param_db)

    # ---- metadata ------------------------------------------------- #

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.constraint]

    # ---- setup ---------------------------------------------------- #

    def setup_for_pose_stack(
        self,
        pose_stack: Any,
        pbt: Any = None,
        constraint_set: Optional[ConstraintSet] = None,
    ) -> ConstraintSet:
        """Return the user-supplied :class:`ConstraintSet` verbatim.

        Upstream tmol lets the constraint set live on the pose stack
        itself. In this port we accept it as an explicit kwarg and pass
        it through, or fall back to ``pose_stack.constraint_set`` /
        ``ConstraintSet.empty()``. The returned object is the term's
        per-pose-stack state and is threaded back into ``__call__``.
        """
        del pbt
        if constraint_set is not None:
            cs = constraint_set
        else:
            cs = getattr(pose_stack, "constraint_set", None)
            if not isinstance(cs, ConstraintSet):
                cs = ConstraintSet.empty()

        target = np.asarray(cs.target_coords)
        indices = np.asarray(cs.atom_indices)
        k = np.asarray(cs.k)
        tol = np.asarray(cs.tol)
        n_constraints = indices.shape[0] if indices.ndim == 2 else -1
        if target.shape != (n_constraints, 3):
            raise ValueError("constraint target_coords must have shape [n, 3]")
        if indices.shape != (n_constraints, 2):
            raise ValueError("constraint atom_indices must have shape [n, 2]")
        if k.shape != (n_constraints,) or tol.shape != (n_constraints,):
            raise ValueError("constraint k and tol must have shape [n]")
        if not (
            np.isfinite(target).all()
            and np.isfinite(k).all()
            and np.isfinite(tol).all()
        ):
            raise ValueError("constraint parameters must be finite")
        if np.any(k < 0) or np.any(tol < 0):
            raise ValueError("constraint k and tol must be nonnegative")
        if n_constraints:
            shape = pose_stack.coords.shape
            if (
                np.any(indices[:, 0] < 0)
                or np.any(indices[:, 0] >= shape[0])
                or np.any(indices[:, 1] < 0)
                or np.any(indices[:, 1] >= shape[1])
            ):
                raise ValueError("constraint atom index is outside the pose")
        return cs

    # ---- evaluation (jit-friendly hot path) ----------------------- #

    def __call__(
        self,
        pose_stack: Any,
        term_state: ConstraintSet,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Compute the per-pose total of 0.5 k max(0, d - tol)^2 terms.

        Parameters
        ----------
        pose_stack
            Supplies ``coords`` and ``n_poses`` (plus any metadata we
            might look at in the future).
        term_state
            The :class:`ConstraintSet` returned by
            :meth:`setup_for_pose_stack`.
        coords_override
            Optional replacement for ``pose_stack.coords``. This is how
            the minimizer gets gradients w.r.t. atomic coordinates.

        Returns
        -------
        dict
            ``{ScoreType.constraint: jnp.float32[n_poses]}`` -- per-pose
            sum of all constraint penalties touching that pose.
        """
        coords = pose_stack.coords if coords_override is None else coords_override
        # ``coords`` is [n_poses, max_n_atoms_per_pose, 3].
        n_poses = coords.shape[0]

        atom_indices = term_state.atom_indices.astype(jnp.int32)
        target = term_state.target_coords.astype(coords.dtype)
        k = term_state.k.astype(coords.dtype)
        tol = term_state.tol.astype(coords.dtype)

        # Fast-path the empty case: return a zero vector without trying
        # to gather from a zero-length index array (which jax handles
        # correctly, but this keeps the trace tidy).
        if atom_indices.shape[0] == 0:
            return {ScoreType.constraint: jnp.zeros((n_poses,), dtype=coords.dtype)}

        pose_ind = atom_indices[:, 0]
        atom_ind = atom_indices[:, 1]

        # Gather the atom coordinate for each constraint.
        # ``coords[pose_ind, atom_ind]`` is [n_constraints, 3].
        atom_xyz = coords[pose_ind, atom_ind]

        # Euclidean distance to target.
        diff = atom_xyz - target  # [n_constraints, 3]
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-20)

        # Flat-bottom harmonic: zero below tol, 0.5 k (d - tol)^2 above.
        excess = jnp.maximum(dist - tol, 0.0)
        penalty = 0.5 * k * excess * excess  # [n_constraints]

        # Scatter-sum into the per-pose total. ``segment_sum`` is the
        # standard JAX primitive for this and is differentiable w.r.t.
        # its ``data`` argument.
        per_pose = jax.ops.segment_sum(penalty, pose_ind, num_segments=n_poses).astype(
            coords.dtype
        )

        return {ScoreType.constraint: per_pose}


__all__ = ["ConstraintSet", "ConstraintEnergyTerm"]
