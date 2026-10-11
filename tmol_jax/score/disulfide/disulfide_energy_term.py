"""Disulfide energy term — pure-JAX port of :mod:`tmol.score.disulfide`.

For every pose, we identify the Cys residues (three-letter name ``CYS``),
pull their CA / CB / SG atom coordinates from the flattened pose coord
slab, and evaluate :func:`potentials.disulfide_pair_E` on every
Cys-Cys pair whose SG-SG distance is under a 3.5 Å cutoff.

The term lives under a single :class:`~tmol_jax.score.score_types.ScoreType`
(``disulfide``) and follows the :class:`~tmol_jax.score.energy_term.EnergyTerm`
two-stage contract: :meth:`setup_for_pose_stack` precomputes static per-pose
atom-index tables (CPU / numpy work, free of tracers) and ``__call__``
is the jit-safe hot path.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.disulfide.potentials import (
    _params_as_dict,
    disulfide_pair_E,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType

# Upstream uses a 3.5 Å SG-SG cutoff when scanning for disulfide-capable
# cysteine pairs.
_SS_CUTOFF_A = 3.5

# Generic-position placeholder geometry (CA_i, CB_i, SG_i, SG_j, CB_j, CA_j)
# substituted for invalid Cys-Cys pairs so the pair potential's angle /
# dihedral gradients are finite on masked-out slots. No coincident atoms,
# no collinear angle/dihedral triples, non-planar — the exact values are
# irrelevant because the energy of invalid pairs is masked to 0.
_CANONICAL_CYSTINE = jnp.asarray(
    [
        [0.0, 0.0, 0.0],  # CA_i
        [1.5, 0.0, 0.0],  # CB_i
        [2.5, 1.2, 0.3],  # SG_i
        [2.6, 3.2, 1.0],  # SG_j
        [1.6, 4.2, 1.8],  # CB_j
        [0.2, 4.0, 2.5],  # CA_j
    ]
)


class DisulfideEnergyTerm(EnergyTerm):
    """Rosetta beta_nov2016 disulfide term as a pure-JAX EnergyTerm."""

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.disulfide]

    def _cached_params(self) -> dict[str, Any]:
        """Cache the 26 disulfide global scalars converted to float32 jnp.

        They're pure-database constants; rebuilding the dict + ``jnp.asarray``
        wrappers every setup took ~0.5 ms on 1UBQ (where there are no Cys
        to score). One-time cached on the term instance.
        """
        cached = getattr(self, "_params_cache", None)
        if cached is not None:
            return cached
        disulfide_db = self.param_db.scoring.disulfide
        params = _params_as_dict(disulfide_db.global_parameters)
        params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}
        self._params_cache = params
        return params

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def setup_for_pose_stack(self, pose_stack: Any, pbt: Any = None) -> dict[str, Any]:
        """Precompute per-pose CA / CB / SG atom indices for every Cys.

        The resulting state is a small fixed-shape dict of jax arrays:

        * ``ca_idx``, ``cb_idx``, ``sg_idx``:
          ``int32[n_poses, max_n_cys]`` — pose-local flat atom indices.
        * ``is_cys``: ``bool[n_poses, max_n_cys]`` — True for real
          residues, False for padding slots.
        * ``params``: dict of the 26 disulfide globals as ``float32``
          scalars, loaded from :class:`DisulfideDatabase`.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types

        block_type_ind = np.asarray(pose_stack.block_type_ind)
        block_coord_offset = np.asarray(pose_stack.block_coord_offset)
        n_poses, max_n_blocks = block_type_ind.shape

        # Per-block-type lookup: (ca, cb, sg) atom indices, or -1 if this
        # block type is not a cysteine. We use atom names 'CA', 'CB', 'SG'
        # which are the canonical Rosetta/tmol names for cysteine.
        n_types = len(pbt.active_block_types)
        bt_ca = np.full(n_types, -1, dtype=np.int32)
        bt_cb = np.full(n_types, -1, dtype=np.int32)
        bt_sg = np.full(n_types, -1, dtype=np.int32)
        for i, bt in enumerate(pbt.active_block_types):
            if getattr(bt, "name3", None) != "CYS":
                continue
            a2i = getattr(bt, "atom_to_idx", None)
            if a2i is None:
                continue
            if "CA" in a2i and "CB" in a2i and "SG" in a2i:
                bt_ca[i] = int(a2i["CA"])
                bt_cb[i] = int(a2i["CB"])
                bt_sg[i] = int(a2i["SG"])

        # Gather per-block: is this a scorable cysteine?
        safe_bti = np.where(block_type_ind >= 0, block_type_ind, 0)
        ca_off = bt_ca[safe_bti]
        cb_off = bt_cb[safe_bti]
        sg_off = bt_sg[safe_bti]
        block_is_cys = (
            (block_type_ind >= 0) & (ca_off >= 0) & (cb_off >= 0) & (sg_off >= 0)
        )

        # Turn block-level offsets into pose-local flat atom indices.
        flat_ca = block_coord_offset + ca_off
        flat_cb = block_coord_offset + cb_off
        flat_sg = block_coord_offset + sg_off

        # Compact to [n_poses, max_n_cys_per_pose].
        n_cys_per_pose = block_is_cys.sum(axis=1)
        max_n_cys = int(n_cys_per_pose.max()) if n_poses > 0 else 0

        ca_idx = np.zeros((n_poses, max_n_cys), dtype=np.int32)
        cb_idx = np.zeros((n_poses, max_n_cys), dtype=np.int32)
        sg_idx = np.zeros((n_poses, max_n_cys), dtype=np.int32)
        is_cys = np.zeros((n_poses, max_n_cys), dtype=bool)
        for p in range(n_poses):
            blocks = np.nonzero(block_is_cys[p])[0]
            k = len(blocks)
            ca_idx[p, :k] = flat_ca[p, blocks]
            cb_idx[p, :k] = flat_cb[p, blocks]
            sg_idx[p, :k] = flat_sg[p, blocks]
            is_cys[p, :k] = True

        # Globals from the scoring DB (cached on the term instance — the
        # ~26 scalars are pure-DB constants, and rebuilding the dict +
        # jnp.asarray wrappers every setup was ~0.5 ms by itself).
        params = self._cached_params()

        return dict(
            ca_idx=jnp.asarray(ca_idx),
            cb_idx=jnp.asarray(cb_idx),
            sg_idx=jnp.asarray(sg_idx),
            is_cys=jnp.asarray(is_cys),
            params=params,
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        pose_stack: Any,
        term_state: dict[str, Any],
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        coords = (
            coords_override if coords_override is not None else pose_stack.coords
        )  # [P, A, 3]

        ca_idx = term_state["ca_idx"]  # [P, C]
        cb_idx = term_state["cb_idx"]
        sg_idx = term_state["sg_idx"]
        is_cys = term_state["is_cys"]
        params = term_state["params"]

        # No cysteines anywhere -> constant zero.
        if ca_idx.shape[1] == 0:
            return {
                ScoreType.disulfide: jnp.zeros((coords.shape[0],), dtype=jnp.float32)
            }

        # Gather per-cys CA/CB/SG coordinates: [P, C, 3].
        ca = jnp.take_along_axis(coords, ca_idx[..., None].repeat(3, axis=-1), axis=1)
        cb = jnp.take_along_axis(coords, cb_idx[..., None].repeat(3, axis=-1), axis=1)
        sg = jnp.take_along_axis(coords, sg_idx[..., None].repeat(3, axis=-1), axis=1)

        # Outer pairs -> [P, C, C, 3] via broadcasting.
        ca_i, ca_j = ca[:, :, None, :], ca[:, None, :, :]
        cb_i, cb_j = cb[:, :, None, :], cb[:, None, :, :]
        sg_i, sg_j = sg[:, :, None, :], sg[:, None, :, :]

        # Valid pairs: both are real Cys, upper-triangle only, SG-SG < cutoff.
        # Safe norm (not jnp.linalg.norm): the i==j diagonal is the zero
        # vector, where norm's Jacobian is 0/0 = NaN. Even though ss_dist
        # only feeds the boolean ``< cutoff`` below, reverse-mode multiplies
        # that NaN by a zero cotangent (NaN*0 = NaN) and leaks it into the
        # coordinate gradient. The +eps under the sqrt keeps the gradient
        # finite and does not change the cutoff test.
        ss_dist = jnp.sqrt(jnp.sum((sg_i - sg_j) ** 2, axis=-1) + 1e-12)
        triu = jnp.triu(jnp.ones((ca_idx.shape[1], ca_idx.shape[1]), dtype=bool), k=1)
        valid = (
            is_cys[:, :, None] & is_cys[:, None, :] & triu & (ss_dist < _SS_CUTOFF_A)
        )

        # Substitute a CONSTANT non-degenerate geometry for all six atoms
        # of invalid pairs. The value is masked to 0 below, but that alone
        # leaks NaN gradients: the potential's internal angle/dihedral ops
        # (``pt_interior_angle`` -> acos, ``dihedral_angle``) have singular
        # derivatives at collinear configurations, and reverse-mode then
        # computes ``0 * NaN = NaN`` on the masked slot. The previous dummy
        # offset the *j* atoms from the real *i* atoms, so masked pairs
        # still depended on real coords through a possibly-collinear
        # geometry and NaN-leaked under jit (eager happened to avoid it).
        # Making every atom of an invalid pair a CONSTANT breaks that
        # dependence entirely: the potential's gradient w.r.t. the real
        # coords is exactly 0 there (constant in -> 0 out), so no NaN. The
        # constants below are an arbitrary generic-position cystine (no
        # coincident atoms, no collinear angle/dihedral triples, non-
        # planar); their numeric value is irrelevant since ``valid`` masks
        # the energy contribution to 0.
        safe = valid[..., None]
        C = _CANONICAL_CYSTINE.astype(coords.dtype)  # [6, 3]
        ca_i_s = jnp.where(safe, ca_i, C[0])
        cb_i_s = jnp.where(safe, cb_i, C[1])
        sg_i_s = jnp.where(safe, sg_i, C[2])
        sg_j_s = jnp.where(safe, sg_j, C[3])
        cb_j_s = jnp.where(safe, cb_j, C[4])
        ca_j_s = jnp.where(safe, ca_j, C[5])

        pair_E = disulfide_pair_E(
            ca_i_s, cb_i_s, sg_i_s, ca_j_s, cb_j_s, sg_j_s, params
        )
        per_pose = jnp.sum(jnp.where(valid, pair_E, 0.0), axis=(1, 2))
        return {ScoreType.disulfide: per_pose.astype(jnp.float32)}


__all__ = ["DisulfideEnergyTerm"]
