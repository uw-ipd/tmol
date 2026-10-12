"""Backbone-dependent omega scoring in pure JAX.

Upstream tmol's omega term evaluates a Gaussian log-probability on
``omega`` centered at a backbone-dependent mean ``mu(phi, psi)`` with
standard deviation ``sigma(phi, psi)`` — both evaluated with a
degree-3 periodic B-spline over 2-D tables indexed by residue type.
Per-residue the energy is

    V = log(sigma / 6) + offset^2 / (2 * sigma^2),

where ``offset = omega_deg - mu`` (wrapped to ``[-180, 180]``) and
both ``mu`` and ``sigma`` are stored in degrees. All arithmetic is
performed *after* interpolation, so the two B-splined channels are the
raw ``mu`` / ``sigma`` grids, not any composite like
``1 / (2 * sigma^2)`` — that keeps the result bit-for-bit compatible
with the upstream kernel.

The cis branch: upstream falls back to a simple non-bb-dep harmonic
``K * omega_rad^2`` with ``K = 32.8 kcal/mol/rad^2`` whenever
``|omega| < pi/2``. We follow the same convention.

We expose:

* :func:`omega_score` — scalar evaluator that accepts a packed
  parameter pytree.
* :class:`OmegaBBDepDatabaseAccess` — helper that wraps
  :class:`~tmol_jax.database.scoring.omega_bbdep.OmegaBBDepDatabase` and
  produces the pre-filtered coefficient tables.

The packed ``OmegaParams`` object is the single argument
:func:`omega_score` expects; this keeps its signature compatible with
per-residue :func:`jax.vmap` dispatch in the whole-pose energy term.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.numeric.bspline import bspline_coefficients
from tmol_jax.score.backbone_torsion._aa import (
    backbone_table_reference_aliases,
    canonical_aa_3letter,
)
from tmol_jax.database.scoring.omega_bbdep import OmegaBBDepDatabase
from tmol_jax.score.backbone_torsion.potentials import bicubic_bspline_interp
from tmol_jax.types import Array, pytree_dataclass

# Default grid shape used when no omega_bbdep.zip is available.
_DEFAULT_N_PHI = 36
_DEFAULT_N_PSI = 36

# Upstream hard-codes the cis-omega spring constant to
# ``omega_V_dV(omega_coords, 32.8)`` (see
# ``tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.impl.hh``).
# Units: kcal/mol/rad^2.
_CIS_OMEGA_K = 32.8


@pytree_dataclass
class OmegaParams:
    """JAX-friendly packed omega_bbdep parameter set.

    Field naming is kept as ``k_table`` / ``omega0_table`` for
    backwards compatibility with the existing
    :class:`~tmol_jax.score.backbone_torsion.backbone_torsion_energy_term.BackboneTorsionTermState`
    pytree layout, but semantically they are pre-filtered B-spline
    coefficients for ``sigma(phi, psi)`` and ``mu(phi, psi)``
    respectively (both in degrees, as upstream stores them).

    Attributes
    ----------
    k_table
        ``float[n_aa, n_phi, n_psi]`` pre-filtered B-spline
        coefficients for the per-AA ``sigma(phi, psi)`` grid, in
        degrees.
    omega0_table
        ``float[n_aa, n_phi, n_psi]`` pre-filtered B-spline
        coefficients for the per-AA ``mu(phi, psi)`` grid, in degrees.
    baseline_table
        Unused. Retained as an all-zero placeholder for backwards
        compatibility; the ``log(sigma/6)`` normalization is computed
        from the interpolated ``sigma`` in :func:`omega_score`.
    bbstart
        ``float[2]`` axis start (radians).
    bbstep
        ``float[2]`` axis step (radians).
    """

    k_table: jax.Array
    omega0_table: jax.Array
    baseline_table: jax.Array
    bbstart: jax.Array
    bbstep: jax.Array


def _wrap_offset_deg(offset_deg: Array) -> Array:
    """Wrap a degree offset into ``[-180, 180]``.

    Upstream only adjusts when ``offset < -180`` (mu is always near
    +180, so ``offset`` lives in ``[-360, +0]``). A generic two-sided
    wrap via mod is equivalent and avoids the data-dependent branch.
    """
    return jnp.mod(offset_deg + 180.0, 360.0) - 180.0


def omega_score(
    phi: Array,
    psi: Array,
    omega: Array,
    aa_type: Array,
    omega_params: OmegaParams,
) -> Array:
    """Scalar backbone-dependent omega energy.

    Matches upstream tmol's ``omega_bbdep_V_dV`` kernel:

    * If ``|omega| < pi/2`` (cis peptide), evaluate a non-bb-dep
      harmonic ``K * omega_rad^2`` with ``K = 32.8``.
    * Otherwise, interpolate ``mu(phi, psi)`` and ``sigma(phi, psi)``
      as degree-3 periodic B-splines, form ``offset_deg = omega_deg -
      mu`` (wrapped to ``[-180, 180]``), and return

          log(sigma / 6) + offset_deg^2 / (2 * sigma^2).

    All Gaussian arithmetic is in degrees — including the
    ``log(sigma/6)`` normalization — so the result matches upstream
    bit-for-bit.

    Parameters
    ----------
    phi, psi, omega
        Scalar JAX arrays (radians). Any dihedral convention works;
        ``phi``/``psi`` are wrapped to the table domain internally.
    aa_type
        Scalar integer index into the leading axis of the packed
        ``k_table`` / ``omega0_table``. Negative values zero the
        result (masks residues with no tabulated omega parameters).
    omega_params
        :class:`OmegaParams` bundle of pre-filtered tables + axis
        metadata.

    Returns
    -------
    jnp.ndarray
        Scalar energy. Zero when ``aa_type < 0`` or when the packed
        tables are empty.
    """
    if omega_params.k_table.shape[0] == 0:
        return jnp.zeros_like(omega)

    # Safe gather for negative aa_type sentinels; mask out at the end.
    safe_idx = jnp.where(aa_type >= 0, aa_type, 0)
    sigma_grid = omega_params.k_table[safe_idx]
    mu_grid = omega_params.omega0_table[safe_idx]

    # Trans branch: bb-dep Gaussian. Interpolate the raw sigma/mu
    # grids (degrees) with a degree-3 B-spline, then compute the
    # Gaussian energy post-interp — same order of operations as
    # upstream's ``omega_bbdep_V_dV``.
    sigma_deg = bicubic_bspline_interp(
        sigma_grid, phi, psi, omega_params.bbstart, omega_params.bbstep
    )
    mu_deg = bicubic_bspline_interp(
        mu_grid, phi, psi, omega_params.bbstart, omega_params.bbstep
    )

    # Guard against zero sigma arising from pathological B-spline
    # overshoot. 1e-6 is below any physical sigma seen in the shipped
    # tables (they bottom out around 3 degrees).
    sigma_safe = jnp.where(sigma_deg > 1e-6, sigma_deg, 1e-6)

    # Upstream: offset = omega_rad * 180/pi - mu_deg; then
    #   "if offset < -180: offset += 360"
    # since mu ~ +180 deg and omega_rad in [-pi, pi] => offset in
    # [-360, 0]. We generalise with a two-sided wrap, which agrees on
    # the data-realisable range.
    omega_deg = omega * (180.0 / jnp.pi)
    offset_deg = _wrap_offset_deg(omega_deg - mu_deg)

    baseline = jnp.log(sigma_safe / 6.0)
    logprob = offset_deg * offset_deg / (2.0 * sigma_safe * sigma_safe)
    v_trans = baseline + logprob

    # Cis branch: non-bb-dep harmonic K * omega_rad^2 with K=32.8.
    # (omega is already in radians.)
    v_cis = _CIS_OMEGA_K * omega * omega

    # Pick the branch. The cis predicate matches upstream's
    # ``omegaang.V > -pi/2 && omegaang.V < pi/2``; evaluating both and
    # selecting is jit-friendly.
    half_pi = 0.5 * jnp.pi
    is_cis = jnp.logical_and(omega > -half_pi, omega < half_pi)
    energy = jnp.where(is_cis, v_cis, v_trans)

    mask = (aa_type >= 0).astype(energy.dtype)
    return energy * mask


class OmegaBBDepDatabaseAccess:
    """Pack an :class:`OmegaBBDepDatabase` into a JAX-ready
    :class:`OmegaParams`.

    Each AA's ``mu(phi, psi)`` and ``sigma(phi, psi)`` grid (degrees)
    is pre-filtered into degree-3 periodic B-spline coefficients via
    :func:`tmol_jax.numeric.bspline.bspline_coefficients` at load
    time. The harmonic spring constant and Gaussian normalization are
    computed at evaluation time from the interpolated values (matching
    upstream's order of operations).

    Parameters
    ----------
    omega_database
        Loaded :class:`OmegaBBDepDatabase`. If the zip payload is
        unavailable the wrapper degrades gracefully: :attr:`params`
        carries empty tables and :attr:`has_tables` is ``False``.

    Attributes
    ----------
    params
        :class:`OmegaParams` pytree — feed directly to
        :func:`omega_score`.
    aa_to_table_idx
        ``np.ndarray[int32]`` per-canonical-AA index into the leading
        axis of the packed tables — non-prepro context only. Retained
        for backwards compatibility with callers that don't consult
        the next-residue identity.
    aa_to_table_idx_2
        ``np.ndarray[int32, shape=(n_canonical_aa, 2)]`` — column 0 is
        the non-prepro table index, column 1 is the prepro table index
        (used when the next residue is proline). Mirrors upstream
        tmol's ``block_type_omega_table[bt][upper_nbr_is_pro]`` access
        pattern.
    has_tables
        ``True`` iff the zip payload loaded successfully.
    """

    __slots__ = (
        "params",
        "aa_to_table_idx",
        "aa_to_table_idx_2",
        "reference_to_table_idx_2",
        "has_tables",
        "_source_uniq_id",
    )

    def __init__(self, omega_database: OmegaBBDepDatabase):
        # Split lookup rows into two keyed dicts: one for the
        # "non-prepro" (upper residue != PRO) context and one for the
        # "prepro" (upper residue == PRO) context. The 2xN mapping is
        # what upstream's ``block_type_omega_table[block_type1][upper_nbr_is_pro]``
        # indexing models.
        name_to_tid_nonprepro: dict[str, str] = {}
        name_to_tid_prepro: dict[str, str] = {}
        for row in omega_database.bbdep_omega_lookup:
            if row.res_upper == "PRO":
                name_to_tid_prepro[row.res_middle] = row.table_id
            else:
                # "_" or anything else is the non-prepro context.
                name_to_tid_nonprepro[row.res_middle] = row.table_id

        table_id_to_idx: dict[str, int] = {}
        sigma_np: list[np.ndarray] = []
        mu_np: list[np.ndarray] = []
        bbstart_np: Optional[np.ndarray] = None
        bbstep_np: Optional[np.ndarray] = None

        for t in omega_database.bbdep_omega_tables:
            if t.table_id in table_id_to_idx:
                continue
            table_id_to_idx[t.table_id] = len(sigma_np)

            # mu and sigma ship in degrees; keep them in degrees so the
            # post-interpolation arithmetic matches upstream directly.
            mu_deg = np.asarray(t.mu, dtype=np.float32)
            sigma_deg = np.asarray(t.sigma, dtype=np.float32)
            # Avoid zero-sigma pathologies (none in the current tables,
            # but the bspline prefilter would blow up if one ever appeared).
            sigma_deg = np.where(sigma_deg > 1e-6, sigma_deg, 1e-6)

            sigma_np.append(sigma_deg)
            mu_np.append(mu_deg)

            if bbstart_np is None:
                bbstart_np = np.asarray(t.bbstart, dtype=np.float32)
                bbstep_np = np.asarray(t.bbstep, dtype=np.float32)

        n_aa = len(canonical_aa_3letter)
        # Column 0: non-prepro table, column 1: prepro table.
        aa_to_table_idx_2 = np.full((n_aa, 2), -1, dtype=np.int32)

        if sigma_np:
            sigma_stacked = np.stack(sigma_np, axis=0).astype(np.float32)
            mu_stacked = np.stack(mu_np, axis=0).astype(np.float32)
            for i, aa in enumerate(canonical_aa_3letter):
                tid_nonprepro = name_to_tid_nonprepro.get(aa)
                tid_prepro = name_to_tid_prepro.get(aa)
                if tid_nonprepro is not None:
                    j = table_id_to_idx.get(tid_nonprepro)
                    if j is not None:
                        aa_to_table_idx_2[i, 0] = j
                if tid_prepro is not None:
                    j = table_id_to_idx.get(tid_prepro)
                    if j is not None:
                        aa_to_table_idx_2[i, 1] = j
            # Back-compat: expose a 1-D non-prepro view as the primary
            # ``aa_to_table_idx`` (callers that don't know about the
            # prepro split still see the non-prepro mapping).
            aa_to_table_idx = aa_to_table_idx_2[:, 0].copy()
            # Pre-filter each AA plane into B-spline coefficients via
            # the separable causal / anti-causal IIR filter (Thevenaz-
            # Blu-Unser, periodic). We vmap over the leading AA axis so
            # the returned coefficients match upstream bit-for-bit for
            # both the mu and sigma tables. The underlying sample grid
            # has 36x36 = 1296 points per AA; prefiltering runs once
            # at load time and costs microseconds.
            _prefilter = jax.vmap(lambda plane: bspline_coefficients(plane, degree=3))
            sigma_coeffs = _prefilter(jnp.asarray(sigma_stacked))
            mu_coeffs = _prefilter(jnp.asarray(mu_stacked))
            # baseline_table is now unused (kept as zeros for pytree
            # layout compatibility with the existing term state).
            empty_baseline = jnp.zeros_like(sigma_coeffs)
            self.params = OmegaParams(
                k_table=sigma_coeffs,
                omega0_table=mu_coeffs,
                baseline_table=empty_baseline,
                bbstart=jnp.asarray(bbstart_np),
                bbstep=jnp.asarray(bbstep_np),
            )
            self.has_tables = True
        else:
            empty = jnp.zeros((0, _DEFAULT_N_PHI, _DEFAULT_N_PSI), dtype=jnp.float32)
            self.params = OmegaParams(
                k_table=empty,
                omega0_table=empty,
                baseline_table=empty,
                bbstart=jnp.array([-jnp.pi, -jnp.pi], dtype=jnp.float32),
                bbstep=jnp.array(
                    [
                        2.0 * jnp.pi / _DEFAULT_N_PHI,
                        2.0 * jnp.pi / _DEFAULT_N_PSI,
                    ],
                    dtype=jnp.float32,
                ),
            )
            self.has_tables = False
            aa_to_table_idx = aa_to_table_idx_2[:, 0].copy()

        references = set(name_to_tid_nonprepro) | set(name_to_tid_prepro)
        self.reference_to_table_idx_2 = {
            reference: np.array(
                [
                    table_id_to_idx.get(name_to_tid_nonprepro.get(reference), -1),
                    table_id_to_idx.get(name_to_tid_prepro.get(reference), -1),
                ],
                dtype=np.int32,
            )
            for reference in references
        }
        for alias, source in backbone_table_reference_aliases.items():
            if source in self.reference_to_table_idx_2:
                self.reference_to_table_idx_2[alias] = self.reference_to_table_idx_2[
                    source
                ].copy()
        self.aa_to_table_idx = aa_to_table_idx
        self.aa_to_table_idx_2 = aa_to_table_idx_2
        self._source_uniq_id = omega_database.uniq_id

    # --------------------------------------------------------------- #
    # Convenience                                                     #
    # --------------------------------------------------------------- #

    @property
    def n_aa(self) -> int:
        return int(self.params.k_table.shape[0])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.params.k_table.shape)  # type: ignore[return-value]

    def table_index_for_aa3(self, aa3: str) -> int:
        try:
            canon_idx = canonical_aa_3letter.index(aa3)
        except ValueError:
            return -1
        return int(self.aa_to_table_idx[canon_idx])


__all__ = [
    "OmegaBBDepDatabaseAccess",
    "OmegaParams",
    "omega_score",
]
