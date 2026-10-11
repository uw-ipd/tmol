"""Reference-energy term — JAX port of :mod:`tmol.score.ref.ref_energy_term`.

Upstream tmol's torch version is a 170-line ``nn.Module`` with a three
stage setup (``setup_block_type`` stashes ``ref_weight`` on every
``RefinedResidueType``, ``setup_packed_block_types`` packs those
per-type weights into a tensor on the ``PackedBlockTypes``, and
``setup_poses`` is a no-op) plus two evaluation entry points
(``eval_ref_energy_for_pose`` / ``eval_ref_energy_for_rotamers``).

The JAX port collapses all of that into the two-method
:class:`~tmol_jax.score.energy_term.EnergyTerm` contract:

* :meth:`setup_for_pose_stack` gathers a ``float32[n_types]`` array of
  ref values indexed by ``PackedBlockTypes`` block-type index. Same as
  upstream's ``packed_block_types.ref_weights`` — we just return it as
  the term state instead of hanging it off the palette. One build per
  palette is enough because every pose stack sharing that palette
  reuses the same index space.

* :meth:`__call__` is a single gather-and-sum: look up the weight for
  every ``(pose, block)`` slot, mask out padding blocks
  (``block_type_ind < 0``) with ``jnp.where``, and sum over blocks to
  produce the ``float32[n_poses]`` channel. No rotamer entry point —
  packing/rotamer evaluation will be handled by the packer (future
  work) calling into this same machinery on a flattened pose stack.

The term is a 1-body energy: there is no cross-block interaction, no
count-pair filter, no gradient flow (``jax.grad`` through ``__call__``
w.r.t. coords returns zero), and no parameter fit — the values come
straight from ``ref.yaml`` and are treated as frozen constants.

No torch, no CUDA kernels, no autograd hook juggling. Just a gather and
a sum, ready for ``jit``/``vmap``/``scan``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class RefEnergyTerm(EnergyTerm):
    """Per-residue-type constant reference energy.

    The score for a pose is ``sum_b ref_values[block_type_ind[p, b]]``
    over real (non-padding) blocks. The ``ref_values`` vector is built
    once per :class:`PackedBlockTypes` palette by looking up each
    block-type's ``base_name`` (the three-letter residue name) in
    ``param_db.scoring.ref.weights``; residues with no entry get
    ``0.0`` (matches upstream tmol behavior).
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        # Raw ``{residue_name: weight}`` mapping from ``ref.yaml``.
        # Stored by reference; used in ``setup_for_pose_stack``.
        self._ref_weights: dict[str, float] = dict(param_db.scoring.ref.weights)

    # ------------------------------------------------------------------ #
    # Metadata.                                                          #
    # ------------------------------------------------------------------ #

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.ref]

    # ------------------------------------------------------------------ #
    # Setup.                                                             #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> dict[str, jax.Array]:
        """Build the ``[n_restypes]`` ref-value vector for this palette.

        The order matches ``PackedBlockTypes.active_block_types``, so
        ``ref_values[block_type_ind[p, b]]`` is the weight for the
        residue type occupying slot ``b`` of pose ``p``.

        Returns a pytree so it survives ``jit`` / ``vmap`` flattening.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types

        # Per-type weights, gathered from the yaml mapping by base_name.
        # Missing entries default to 0.0 — matches upstream behavior.
        weights_np = np.zeros((pbt.n_types,), dtype=np.float32)
        for i, bt in enumerate(pbt.active_block_types):
            w = self._ref_weights.get(bt.base_name, 0.0)
            weights_np[i] = float(w)

        return {"ref_values": jnp.asarray(weights_np, dtype=jnp.float32)}

    # ------------------------------------------------------------------ #
    # Evaluation (hot path).                                             #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        pose_stack: Any,
        term_state: dict[str, jax.Array],
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Sum per-block ref weights across each pose.

        The reference term is coordinate-independent; ``coords_override``
        is accepted only for EnergyTerm API compatibility with other
        terms that do depend on coords.
        """
        del coords_override  # not used — ref is coord-independent

        ref_values = term_state["ref_values"]  # float32[n_types]
        bti = pose_stack.block_type_ind  # int32[n_poses, max_n_blocks]

        # Gather weights, clamping padding (-1) indices to 0 so the
        # gather is safe; then ``where`` zeros the padding contributions
        # so they never reach the sum.
        real = bti >= 0
        safe_idx = jnp.where(real, bti, 0)
        per_block = ref_values[safe_idx]
        per_block = jnp.where(real, per_block, 0.0).astype(jnp.float32)

        per_pose = jnp.sum(per_block, axis=1).astype(jnp.float32)
        return {ScoreType.ref: per_pose}


__all__ = ["RefEnergyTerm"]
