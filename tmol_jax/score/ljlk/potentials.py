"""Pure-function LJ/LK potentials in JAX.

Line-by-line port of ``tmol/score/ljlk/potentials/lj.hh`` (the shifted
+ faded Lennard-Jones split into attractive / repulsive channels) and
``tmol/score/ljlk/potentials/lk_isotropic.hh`` (symmetrised
Lazaridis-Karplus desolvation).

All branches are resolved via :func:`jax.numpy.where` so these work
under ``jit``/``vmap``/``grad`` and over arrays of any rank. ``dist``
and per-type parameter arrays must broadcast; the typical calling
convention is a ``[n_pairs_i, n_pairs_j]`` distance matrix with
``[n_pairs_i, 1]`` / ``[1, n_pairs_j]`` parameter arrays.

Vectorisation shape contract
---------------------------
Every scalar input (``lj_radius``, ``lj_wdepth``, ``is_donor``, ...)
may have arbitrary leading shape. Fields must broadcast to the same
shape as ``dist``. The output has that same broadcast shape.

The ``i_params`` / ``j_params`` objects must expose the LJTypeParams
attributes (``lj_radius``, ``lj_wdepth``, ``is_donor``, ``is_acceptor``,
``is_hydroxyl``, ``is_polarh``, ``lk_volume``, ``lk_dgfree``,
``lk_lambda``). They are *not* required to be :class:`LJTypeParams`
instances -- any pytree with these fields works (the term calls
``.gather(idx)`` to pack them per-atom-pair).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from tmol_jax.score.common import connectivity_weight, interpolate, interpolate_to_zero

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Far / near cutoff geometry derives from ``global_params.max_dis``
# (upstream commit 1d7e18c). The "spline start" is ``max_dis - 1.5``;
# the linear-extrapolation crossover is ``sigma * lj_dlin_sigma_factor``.
# Only the hard offset between ``cpoly_dmax`` and the cap on
# ``cpoly_dmin`` is still a fixed literal (matches upstream's
# ``Real(0.1)``).
_LJ_CPOLY_DMIN_MAX_OFFSET: float = 0.1

# Width of the far fade (``cpoly_dmax - spline_start``), hard-coded in
# upstream as ``1.5`` A.
_FAR_FADE_WIDTH: float = 1.5

# pi^1.5 -- matches the literal in ``lk_isotropic.hh``.
_PI_POW_1P5: float = 5.56832799683


# --------------------------------------------------------------------------- #
# lj_sigma                                                                     #
# --------------------------------------------------------------------------- #


def lj_sigma(i_params: Any, j_params: Any, global_params: Any) -> jax.Array:
    """Per-pair ``sigma`` with hydrogen-bond overrides.

    Vectorised port of ``tmol::score::ljlk::potentials::lj_sigma`` in
    ``common.hh``. Four cases:

    1. Heavy donor / acceptor (non-hydroxyl):   ``sigma = lj_hbond_dis``.
    2. Hydroxyl donor / acceptor:               ``sigma = lj_hbond_OH_donor_dis``.
    3. Polar-H donor / acceptor:                ``sigma = lj_hbond_hdis``.
    4. Otherwise:                               ``sigma = lj_radius_i + lj_radius_j``.

    Each ``i_params`` / ``j_params`` attribute is expected to broadcast
    against the others. The output has the broadcast shape.
    """
    std_sigma = i_params.lj_radius + j_params.lj_radius

    # Mirror common.hh:29-43 exactly. Outer branch: any donor/acceptor pair.
    # Inside it, the hydroxyl-donor test is a *capability* check on either
    # atom (NOT gated by which side is the acceptor) and it wins -> OH dis.
    # The polar-H/acceptor case is a lower-precedence ``else if``.
    is_don_acc_pair = (i_params.is_donor & j_params.is_acceptor) | (
        j_params.is_donor & i_params.is_acceptor
    )
    is_hydroxyl_donor = (i_params.is_donor & i_params.is_hydroxyl) | (
        j_params.is_donor & j_params.is_hydroxyl
    )
    is_polarh_pair = (i_params.is_polarh & j_params.is_acceptor) | (
        j_params.is_polarh & i_params.is_acceptor
    )

    # ``else if`` polar-H/acceptor (lower precedence than donor/acceptor).
    sigma = jnp.where(is_polarh_pair, global_params.lj_hbond_hdis, std_sigma)
    # ``if`` donor/acceptor: hydroxyl-donor -> OH dis, else -> standard hbond dis.
    donor_acceptor_sigma = jnp.where(
        is_hydroxyl_donor,
        global_params.lj_hbond_OH_donor_dis,
        global_params.lj_hbond_dis,
    )
    sigma = jnp.where(is_don_acc_pair, donor_acceptor_sigma, sigma)
    return sigma


# --------------------------------------------------------------------------- #
# Bare Lennard-Jones (no shift / fade)                                         #
# --------------------------------------------------------------------------- #


def vdw(dist: jax.Array, sigma: jax.Array, epsilon: jax.Array) -> jax.Array:
    """Bare Lennard-Jones potential, no shift/fade.

    ``V = epsilon * (sd^12 - 2 sd^6)`` with ``sd = sigma / dist``.

    NaN-safe under ``jax.grad``: ``dist`` can be 0 on padded / virtual
    pairs (e.g. two atoms at identical coordinates), which would give
    ``sd = sigma/0 = inf`` and propagate into the cotangent. The caller
    always masks this branch out of the forward sum, but ``jnp.where``
    still pulls cotangents through both branches, so we need the
    "double-where" trick: swap ``dist`` for a safe constant inside the
    division, then never look at the output on the masked-off path.
    """
    safe_d = dist > 0.0
    safe_dist = jnp.where(safe_d, dist, 1.0)
    sd = sigma / safe_dist
    sd2 = sd * sd
    sd6 = sd2 * sd2 * sd2
    sd12 = sd6 * sd6
    V = epsilon * (sd12 - 2.0 * sd6)
    # On the zero-dist branch substitute a finite dummy; downstream
    # selectors (``in_vdw`` etc. in ``lj_atr_rep``) zero its cotangent.
    return jnp.where(safe_d, V, jnp.zeros_like(V))


def vdw_and_deriv(
    dist: jax.Array, sigma: jax.Array, epsilon: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Bare LJ value and its derivative w.r.t. ``dist``.

    Returns ``(V, dV/ddist)``. Formula is analytic and matches the C++
    ``vdw<Real>::V_dV`` in ``lj.hh``.

    Note: we hand-roll the derivative (instead of just ``jax.grad``)
    because this expression appears *inside* the cpoly fade branch and
    we need the derivative evaluated at ``cpoly_dmin`` (a per-pair
    quantity) for every pair simultaneously -- a vmap over grad is
    slower than the analytic form.

    NaN-safety: ``dist`` is typically a *constant* (e.g. ``cpoly_dmin``
    or ``d_lin = sigma * 0.6``) and can be 0 when ``sigma == 0`` on a
    padded / virtual pair. The double-where guard below ensures both
    the primal and the cotangent stay finite — even though the caller
    always masks this pair out of the final sum via ``jnp.where``.
    """
    safe_d = dist > 0.0
    safe_dist = jnp.where(safe_d, dist, 1.0)
    sd = sigma / safe_dist
    sd2 = sd * sd
    sd6 = sd2 * sd2 * sd2
    sd12 = sd6 * sd6
    V = epsilon * (sd12 - 2.0 * sd6)
    # d/ddist (sd^n) = -n * sd^n / dist
    dV_ddist = epsilon * ((-12.0 * sd12 / safe_dist) - (-12.0 * sd6 / safe_dist))
    V = jnp.where(safe_d, V, jnp.zeros_like(V))
    dV_ddist = jnp.where(safe_d, dV_ddist, jnp.zeros_like(dV_ddist))
    return V, dV_ddist


# --------------------------------------------------------------------------- #
# LJ (shift + fade)                                                            #
# --------------------------------------------------------------------------- #


def lj_atr_rep(
    dist: jax.Array,
    bonded_path_length: jax.Array,
    i_params: Any,
    j_params: Any,
    global_params: Any,
    soft_core_cap: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Lennard-Jones attractive / repulsive pair with shift + fade.

    Vectorised port of ``tmol::score::ljlk::potentials::lj_score<Real>::V``
    in ``lj.hh``. Returns two broadcast-shaped arrays ``(Vatr, Vrep)``
    already multiplied by the count-pair weight.

    The four piecewise regions (in C++ order) are:

    1. ``dist > cpoly_dmax``  -> ``Vatr = Vrep = 0``.
    2. ``dist > cpoly_dmin`` -> attractive portion faded to zero via
       the cubic Hermite polynomial joining ``vdw(cpoly_dmin)`` to
       ``(0, 0)`` at ``cpoly_dmax``; ``Vrep = 0``.
    3. ``dist > d_lin``      -> ``Vatr = vdw(dist)``.
    4. ``dist <= d_lin``     -> ``Vatr = vdw(d_lin) +
                                        dVdw/ddist(d_lin) * (dist - d_lin)``
       (linear extrapolation into the "hard-wall" region).

    After selecting ``Vatr`` in regions 3/4:

    * If ``dist < sigma``: ``Vrep = Vatr + epsilon`` and ``Vatr = -epsilon``
      (clamp the attractive portion at its minimum of ``-epsilon``).
    * Otherwise: ``Vrep = 0``.

    All branches are expressed with ``jnp.where``; the outputs are
    multiplied by ``connectivity_weight(bonded_path_length)``.
    """
    sigma = lj_sigma(i_params, j_params, global_params)
    # Geometric-mean well depth. Clamp to a tiny positive floor so
    # ``sqrt`` has a finite gradient even when either wdepth is zero
    # (virtual atoms / padding). The downstream mask still nulls these
    # contributions out of the sum.
    epsilon = jnp.sqrt(jnp.maximum(i_params.lj_wdepth * j_params.lj_wdepth, 1e-12))

    cpoly_dmax = jnp.asarray(global_params.max_dis, dtype=sigma.dtype)
    spline_start = cpoly_dmax - jnp.asarray(_FAR_FADE_WIDTH, dtype=sigma.dtype)
    d_lin = sigma * jnp.asarray(global_params.lj_dlin_sigma_factor, dtype=sigma.dtype)
    cpoly_dmin_cap = cpoly_dmax - _LJ_CPOLY_DMIN_MAX_OFFSET
    cpoly_dmin = jnp.where(
        sigma > spline_start,
        jnp.minimum(sigma, cpoly_dmin_cap),
        spline_start,
    )

    # Regions
    above_cutoff = dist > cpoly_dmax
    in_fade = (dist > cpoly_dmin) & (~above_cutoff)
    in_vdw = (dist > d_lin) & (~in_fade) & (~above_cutoff)

    # --- fade branch ---------------------------------------------------
    # Evaluate vdw at cpoly_dmin for every pair (may produce garbage in
    # non-fade pairs but masked out below).
    V_at_dmin, dV_at_dmin = vdw_and_deriv(cpoly_dmin, sigma, epsilon)
    # ``interpolate_to_zero`` returns y==0 at x==x1, so divide-by-zero is
    # avoided by our region mask (``in_fade`` requires dist > cpoly_dmin,
    # which is strictly less than cpoly_dmax).
    V_fade = interpolate_to_zero(dist, cpoly_dmin, V_at_dmin, dV_at_dmin, cpoly_dmax)

    # --- bare vdw branch -----------------------------------------------
    V_bare = vdw(dist, sigma, epsilon)

    # --- linear branch -------------------------------------------------
    V_at_dlin, dV_at_dlin = vdw_and_deriv(d_lin, sigma, epsilon)
    V_lin = V_at_dlin + dV_at_dlin * (dist - d_lin)

    # Combine into ``Vatr_raw`` (before the sigma split)
    Vatr_raw = jnp.where(
        above_cutoff,
        jnp.zeros_like(dist),
        jnp.where(in_fade, V_fade, jnp.where(in_vdw, V_bare, V_lin)),
    )

    # Repulsive / attractive split:
    #   dist < sigma: Vrep = Vatr_raw + epsilon, Vatr = -epsilon
    #   else        : Vrep = 0,                 Vatr = Vatr_raw
    # But ALSO: when above_cutoff, both are zero.
    below_sigma = dist < sigma
    Vatr = jnp.where(below_sigma, -epsilon, Vatr_raw)
    Vrep = jnp.where(below_sigma, Vatr_raw + epsilon, jnp.zeros_like(dist))

    # Zero everything outside the cutoff.
    Vatr = jnp.where(above_cutoff, jnp.zeros_like(dist), Vatr)
    Vrep = jnp.where(above_cutoff, jnp.zeros_like(dist), Vrep)

    # Opt-in soft-core repulsion (OFF by default -> exact upstream parity).
    # A stiff clash makes ``Vrep`` (and its gradient) explode, which
    # destabilises minimisation / fine-tuning through the energy. When a cap
    # is given, smoothly saturate the repulsive channel: ``cap*tanh(Vrep/cap)``
    # is ~identity for ``Vrep << cap`` (near-native pairs unchanged) and
    # plateaus at ``cap`` for deep clashes, so a single overlapping pair can
    # no longer produce an unbounded force. Attractive channel is untouched.
    if soft_core_cap is not None:
        c = jnp.asarray(soft_core_cap, dtype=Vrep.dtype)
        Vrep = c * jnp.tanh(Vrep / c)

    # Count-pair weight
    w = connectivity_weight(bonded_path_length).astype(Vatr.dtype)
    return w * Vatr, w * Vrep


# --------------------------------------------------------------------------- #
# LK isotropic desolvation                                                     #
# --------------------------------------------------------------------------- #


def _f_desolv(
    dist: jax.Array,
    lj_radius_i: jax.Array,
    lk_dgfree_i: jax.Array,
    lk_lambda_i: jax.Array,
    lk_volume_j: jax.Array,
) -> jax.Array:
    """Single-direction LK desolvation.

    ``-v_j * dg_i / (2 * pi^1.5 * lambda_i) / dist^2
        * exp(-((dist - r_i) / lambda_i)^2)``

    NaN-safe: ``lk_lambda_i`` is 0 on padded / virtual atoms (the
    ljlk parameter-gather fills padding with zeros), which would
    divide by zero in both ``z`` and the prefactor. Likewise ``dist``
    can be 0 on collinear pairs. Both are guarded via the double-where
    trick so gradients stay finite through ``jnp.where`` cascades in
    the caller; the caller zeroes the pair contribution downstream.
    """
    safe_lambda = lk_lambda_i != 0.0
    lambda_safe = jnp.where(safe_lambda, lk_lambda_i, 1.0)
    safe_d = dist > 0.0
    dist_safe = jnp.where(safe_d, dist, 1.0)
    z = (dist - lj_radius_i) / lambda_safe
    V = (
        -lk_volume_j
        * lk_dgfree_i
        / (2.0 * _PI_POW_1P5 * lambda_safe)
        / (dist_safe * dist_safe)
        * jnp.exp(-z * z)
    )
    ok = safe_lambda & safe_d
    return jnp.where(ok, V, jnp.zeros_like(V))


def _f_desolv_and_deriv(
    dist: jax.Array,
    lj_radius_i: jax.Array,
    lk_dgfree_i: jax.Array,
    lk_lambda_i: jax.Array,
    lk_volume_j: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """``f_desolv`` value and derivative (from ``lk_isotropic.hh``).

    NaN-safe: see :func:`_f_desolv` — ``lk_lambda_i`` may be 0 for
    virtual atoms and ``dist`` may be 0 on degenerate pairs. Both are
    guarded.
    """
    safe_lambda = lk_lambda_i != 0.0
    lambda_safe = jnp.where(safe_lambda, lk_lambda_i, 1.0)
    safe_d = dist > 0.0
    dist_safe = jnp.where(safe_d, dist, 1.0)
    z = (dist - lj_radius_i) / lambda_safe
    exp_val = jnp.exp(-z * z)
    coeff = -lk_volume_j * lk_dgfree_i / (2.0 * _PI_POW_1P5 * lambda_safe)
    V = coeff * exp_val / (dist_safe * dist_safe)
    dV_ddist = (
        coeff
        * exp_val
        * (
            (-2.0 / (dist_safe * dist_safe * dist_safe))
            + (1.0 / (dist_safe * dist_safe))
            * -(2.0 * dist - 2.0 * lj_radius_i)
            / (lambda_safe * lambda_safe)
        )
    )
    ok = safe_lambda & safe_d
    V = jnp.where(ok, V, jnp.zeros_like(V))
    dV_ddist = jnp.where(ok, dV_ddist, jnp.zeros_like(dV_ddist))
    return V, dV_ddist


def _lk_one_sided_with_fade(
    dist: jax.Array,
    lj_sigma_ij: jax.Array,
    lj_radius_i: jax.Array,
    lk_dgfree_i: jax.Array,
    lk_lambda_i: jax.Array,
    lk_volume_j: jax.Array,
    cpoly_far_dmax: jax.Array,
    cpoly_far_dmin: jax.Array,
    is_cc_pair=False,
) -> jax.Array:
    """One-direction LK with upstream's full close- and far-cpoly fades.

    Faithful port of ``lk_isotropic_pair::V`` in
    ``tmol/score/ljlk/potentials/lk_isotropic.hh`` (one-sided: i -> j).
    Five regions (in upstream order):

    * ``dist > cpoly_far_dmax`` (6.0 A)  -> 0
    * ``dist > cpoly_far_dmin`` (4.5 A)  -> cubic Hermite fade to zero
    * ``dist > cpoly_close_dmax``        -> raw ``f_desolv(dist)``
    * ``dist > cpoly_close_dmin``        -> cubic Hermite ramp from
          ``(cpoly_close_dmin, f_desolv(d_min), 0)`` to
          ``(cpoly_close_dmax, f_desolv(cpoly_close_dmax),
              f'_desolv(cpoly_close_dmax))``
    * else (``dist <= cpoly_close_dmin``) -> flat plateau at
          ``f_desolv(d_min)``.

    Here ``d_min = sigma * 0.89``,
    ``cpoly_close_dmin = sqrt(max(d_min^2 - 1.45, 0.01))``, and
    ``cpoly_close_dmax = sqrt(d_min^2 + 1.05)``.

    ``lj_sigma_ij`` is the pair's effective LJ sigma (with H-bond
    overrides applied) obtained from :func:`lj_sigma`; it is shared
    between the two directions so the caller computes it once.
    """
    d_min = jnp.where(
        is_cc_pair, jnp.maximum(lj_sigma_ij * 0.89, 4.2), lj_sigma_ij * 0.89
    )
    # Canonical etable knots lie on bins spaced 1/20 Angstrom squared.
    n = jnp.floor(20 * d_min * d_min)
    cpoly_close_dmin = jnp.sqrt(jnp.maximum(0, n - 29) / 20)
    cpoly_close_dmax = jnp.sqrt(jnp.minimum(n + 21, 405) / 20)

    # --- Far fade ------------------------------------------------------
    V_far_min, dV_far_min = _f_desolv_and_deriv(
        cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
    )
    V_far_fade = interpolate_to_zero(
        dist, cpoly_far_dmin, V_far_min, dV_far_min, cpoly_far_dmax
    )

    # --- Close plateau & close fade -----------------------------------
    # f_desolv at d_min (used as the plateau value and as the low
    # endpoint of the close fade).
    V_at_dmin = _f_desolv(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    # f_desolv and its derivative at cpoly_close_dmax (upper endpoint
    # of the close fade).
    V_close_max, dV_close_max = _f_desolv_and_deriv(
        cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
    )
    # Cubic Hermite from (cpoly_close_dmin, V_at_dmin, 0) to
    # (cpoly_close_dmax, V_close_max, dV_close_max).
    V_close_fade = interpolate(
        dist,
        cpoly_close_dmin,
        V_at_dmin,
        jnp.zeros_like(V_at_dmin),
        cpoly_close_dmax,
        V_close_max,
        dV_close_max,
    )

    # --- Bare f_desolv ------------------------------------------------
    V_bare = _f_desolv(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)

    # Nested ``where`` mirroring upstream's if/else-if cascade, evaluated
    # outside-in (tightest cutoff last):
    #   above far_dmax -> 0
    #   above far_dmin -> V_far_fade
    #   above close_dmax -> V_bare
    #   above close_dmin -> V_close_fade
    #   else             -> V_at_dmin
    V = jnp.where(dist > cpoly_close_dmin, V_close_fade, V_at_dmin)
    V = jnp.where(dist > cpoly_close_dmax, V_bare, V)
    V = jnp.where(dist > cpoly_far_dmin, V_far_fade, V)
    V = jnp.where(dist > cpoly_far_dmax, jnp.zeros_like(dist), V)
    return V


def lk_isotropic_pair(
    dist: jax.Array,
    bonded_path_length: jax.Array,
    i_params: Any,
    j_params: Any,
    global_params: Any,
) -> jax.Array:
    """Symmetrised Lazaridis-Karplus isotropic desolvation.

    Computes ``f_desolv(i -> j) + f_desolv(j -> i)`` with upstream's
    piecewise close- and far-range cubic Hermite fades applied to each
    direction, and multiplied by the count-pair weight.

    See ``tmol/score/ljlk/potentials/lk_isotropic.hh`` for the C++
    reference. The close-range cpoly runs between
    ``cpoly_close_dmin = sqrt(max((sigma*0.89)^2 - 1.45, 0.01))`` and
    ``cpoly_close_dmax = sqrt((sigma*0.89)^2 + 1.05)``; below
    ``cpoly_close_dmin`` the LK contribution is clamped to the plateau
    value ``f_desolv(sigma*0.89)``. The far-range cpoly runs between
    4.5 and 6.0 A and fades to zero.
    """
    # Shared across the two directions (sigma is symmetric in i/j).
    sigma = lj_sigma(i_params, j_params, global_params)
    cpoly_far_dmax = jnp.asarray(global_params.max_dis, dtype=dist.dtype)
    cpoly_far_dmin = cpoly_far_dmax - jnp.asarray(_FAR_FADE_WIDTH, dtype=dist.dtype)

    V_ij = _lk_one_sided_with_fade(
        dist,
        sigma,
        i_params.lj_radius,
        i_params.lk_dgfree,
        i_params.lk_lambda,
        j_params.lk_volume,
        cpoly_far_dmax,
        cpoly_far_dmin,
        i_params.is_carbon_lk & j_params.is_carbon_lk,
    )
    V_ji = _lk_one_sided_with_fade(
        dist,
        sigma,
        j_params.lj_radius,
        j_params.lk_dgfree,
        j_params.lk_lambda,
        i_params.lk_volume,
        cpoly_far_dmax,
        cpoly_far_dmin,
        i_params.is_carbon_lk & j_params.is_carbon_lk,
    )
    V = V_ij + V_ji

    w = connectivity_weight(bonded_path_length).astype(V.dtype)
    return w * V


__all__ = [
    "lj_sigma",
    "vdw",
    "vdw_and_deriv",
    "lj_atr_rep",
    "lk_isotropic_pair",
]
