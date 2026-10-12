"""Hydrogen-bond energy term for tmol_jax.

Port of :class:`tmol.score.hbond.hbond_energy_term.HBondEnergyTerm` with
a pure-JAX, vectorised inner kernel. The upstream implementation is a
thin wrapper around a compiled CUDA / CPU tile-scoring kernel; here we
evaluate every donor-acceptor pair in parallel using broadcasting and
let XLA handle the blocking on-device.

The evaluation strategy:

1. Build :class:`HBondParams` once per ``PoseStack`` via
   :func:`tmol_jax.score.hbond.params.hbond_params_for_pose_stack`.
2. For each pose, the donor atoms live in a ``[P, D]`` index array and
   the acceptor atoms in a ``[P, A]`` index array.
3. The pair energies form a ``[P, D, A]`` grid. We gather all 5 atom
   coordinates (D, H, A, B, B0) per (pose, donor, acceptor) slot,
   gather the matching polynomial / weight params by (donor_type,
   acceptor_type), and call :func:`hbond_pair_V` once on the full grid.
4. Mask out "padding" pairs (either the donor index or the acceptor
   index is -1) and pairs whose A-H distance exceeds ``max_ah_dist``,
   then sum.

The :class:`HBondEnergyTerm` class itself implements the
:class:`tmol_jax.score.energy_term.EnergyTerm` contract:

* :meth:`setup_for_pose_stack` returns an :class:`HBondParams` pytree.
* :meth:`__call__` runs the pair grid under ``jit``/``grad`` and
  accepts a ``coords_override`` so the minimizer can differentiate
  through the coordinate array directly.

Returns ``{ScoreType.hbond: [n_poses]}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from tmol_jax.score.common import distance
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.hbond.params import HBondParams, hbond_params_for_pose_stack
from tmol_jax.score.hbond.potentials import hbond_pair_V
from tmol_jax.score.score_types import ScoreType

# ---------------------------------------------------------------------- #
# Gather helpers                                                         #
# ---------------------------------------------------------------------- #


def _gather_xyz(coords_P: jnp.ndarray, atom_idx: jnp.ndarray) -> jnp.ndarray:
    """Gather per-pose coords by index.

    ``coords_P`` is ``[P, N_atoms, 3]``. ``atom_idx`` is ``[P, K]``,
    with ``-1`` as the padding sentinel. Padding positions gather from
    index 0 and are masked out downstream by the caller.
    """
    safe_idx = jnp.where(atom_idx >= 0, atom_idx, 0)
    # Broadcast the index to [P, K, 3] so the gather hits all three
    # coord components in one shot.
    idx_bc = jnp.broadcast_to(safe_idx[..., None], safe_idx.shape + (3,))
    return jnp.take_along_axis(coords_P, idx_bc, axis=1)


def _pair_gather(
    table: jnp.ndarray, donor_type: jnp.ndarray, acceptor_type: jnp.ndarray
) -> jnp.ndarray:
    """Outer-product gather from a ``[n_don_types, n_acc_types, ...]`` table.

    ``donor_type`` is ``[P, D]`` and ``acceptor_type`` is ``[P, A]``;
    the result is ``[P, D, A, ...]``. Implemented with broadcasted
    advanced indexing.
    """
    D = donor_type[:, :, None]  # [P, D, 1]
    A = acceptor_type[:, None, :]  # [P, 1, A]
    D_b, A_b = jnp.broadcast_arrays(D, A)
    return table[D_b, A_b]


# ---------------------------------------------------------------------- #
# Per-pose scoring                                                       #
# ---------------------------------------------------------------------- #


def hbond_pose_scores(coords: jnp.ndarray, params: HBondParams) -> jnp.ndarray:
    """Compute total hbond energy for each pose from raw coords.

    Parameters
    ----------
    coords:
        ``[n_poses, max_n_atoms_per_pose, 3]`` coordinate tensor. This
        is typically ``pose_stack.coords`` or a ``coords_override``
        supplied by the minimizer.
    params:
        Pre-materialised :class:`HBondParams` for the pose stack.

    Returns
    -------
    ``float32[n_poses]`` array of total hbond energies.
    """
    ap = params.atom_params
    pp = params.pair_params
    polys = params.polynomials
    gp = params.global_params

    # Coordinate gathers: expand donor H, donor heavy, acc, acc base,
    # acc base0 into [P, D/A, 3] arrays.
    H_xyz = _gather_xyz(coords, ap.donor_H_atom)  # [P, D, 3]
    D_xyz = _gather_xyz(coords, ap.donor_heavy_atom)  # [P, D, 3]
    A_xyz = _gather_xyz(coords, ap.acceptor_atom)  # [P, A, 3]
    B_xyz = _gather_xyz(coords, ap.acceptor_base)  # [P, A, 3]
    B0_xyz = _gather_xyz(coords, ap.acceptor_base0)  # [P, A, 3]

    # Broadcast to [P, D, A, 3] so every pair gets its 5 coords.
    D_g = D_xyz[:, :, None, :]
    H_g = H_xyz[:, :, None, :]
    A_g = A_xyz[:, None, :, :]
    B_g = B_xyz[:, None, :, :]
    B0_g = B0_xyz[:, None, :, :]

    # Pair-param gathers: [P, D, A, ...] grids of weight / hybridization /
    # packed-polynomial vectors.
    donor_type = ap.donor_type  # [P, D]
    acceptor_type = ap.acceptor_type  # [P, A]
    weight_grid = _pair_gather(pp.weight, donor_type, acceptor_type)  # [P, D, A]
    hyb_grid = _pair_gather(pp.acceptor_hybridization, donor_type, acceptor_type)
    ahdist_grid = _pair_gather(
        polys.AHdist_poly, donor_type, acceptor_type
    )  # [P, D, A, 15]
    ahd_grid = _pair_gather(polys.cosAHD_poly, donor_type, acceptor_type)
    bah_grid = _pair_gather(polys.cosBAH_poly, donor_type, acceptor_type)

    # Compute raw pair energy over the [P, D, A] grid.
    E_grid = hbond_pair_V(
        D_g,
        H_g,
        A_g,
        B_g,
        B0_g,
        hyb_grid,
        weight_grid,
        ahdist_grid,
        ahd_grid,
        bah_grid,
        gp,
    )

    # Padding + cutoff mask. Use the NaN-safe ``distance`` helper so the
    # cutoff test stays gradient-safe when two atoms collide (no physical
    # hbond can happen there anyway, but grads still need to be finite
    # for XLA to propagate through the ``where``).
    valid_don = (ap.donor_H_atom >= 0)[:, :, None]  # [P, D, 1]
    valid_acc = (ap.acceptor_atom >= 0)[:, None, :]  # [P, 1, A]
    valid_pair = valid_don & valid_acc

    ah_dist = distance(A_g, H_g)  # [P, D, A]
    within_cutoff = ah_dist < gp.max_ah_dist

    # Count-pair exclusion: skip pairs with fewer than 5 bonds between
    # the donor-H and the acceptor atom. Matches upstream's
    # ``if (cp_separation < 5) return 0`` guard.
    cp_sep = params.donor_acceptor_bond_sep  # [P, D, A] int32
    cp_ok = cp_sep >= 5

    mask = valid_pair & within_cutoff & cp_ok
    E_grid = jnp.where(mask, E_grid, jnp.zeros_like(E_grid))

    return jnp.sum(E_grid, axis=(1, 2))


# ---------------------------------------------------------------------- #
# EnergyTerm subclass                                                    #
# ---------------------------------------------------------------------- #


class HBondEnergyTerm(EnergyTerm):
    """Hydrogen-bond parameter setup and pair scoring."""

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        self.hbond_db = param_db.scoring.hbond
        self.chemical_db = param_db.chemical

    # ------------------------------------------------------------------ #
    # Metadata.                                                          #
    # ------------------------------------------------------------------ #

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.hbond]

    # ------------------------------------------------------------------ #
    # Setup.                                                             #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Optional[Any] = None
    ) -> HBondParams:
        """Build per-pose :class:`HBondParams` for ``pose_stack``.

        The returned pytree bundles donor/acceptor atom indices,
        per-pair weights, and polynomial tables. Threaded into
        :meth:`__call__` via the ``term_state`` argument.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        return hbond_params_for_pose_stack(
            pose_stack,
            pbt,
            self.hbond_db,
            self.chemical_db,
        )

    # ------------------------------------------------------------------ #
    # Evaluation (hot path).                                             #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        pose_stack: Any,
        term_state: HBondParams,
        coords_override: Optional[jax.Array] = None,
    ) -> Dict[ScoreType, jnp.ndarray]:
        """Return ``{ScoreType.hbond: [n_poses]}`` unweighted energies.

        The ``coords_override`` argument replaces ``pose_stack.coords``
        for this evaluation only — the block-type / donor-acceptor
        metadata in ``term_state`` still reflects the underlying pose.
        This lets ``jax.grad`` trace straight through the coord array
        without reconstructing the :class:`PoseStack` pytree.
        """
        coords = coords_override if coords_override is not None else pose_stack.coords
        if coords.dtype == jnp.float32 and jax.default_backend() == "gpu":
            from tmol_jax.score.hbond import cuda

            if cuda.available():
                return {ScoreType.hbond: cuda.score(term_state, coords)}

        energies = hbond_pose_scores(coords, term_state)
        return {ScoreType.hbond: energies}


__all__ = [
    "HBondEnergyTerm",
    "hbond_pose_scores",
]
