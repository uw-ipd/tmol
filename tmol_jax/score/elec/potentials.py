"""Sigmoidal-dielectric Coulombic potential, pure JAX.

Direct port of :file:`tmol/score/elec/potentials/potentials.hh`.

The upstream ``elec(...)`` routine builds a piecewise energy profile:

    dist < min_dis - 0.25    -> flat at min_dis
    dist < min_dis + 0.25    -> cubic-Hermite short-range fade
    dist < max_dis - 1.0     -> true Coulombic   eiej * (C1/(d*eps(d)) - C2)
    dist < max_dis           -> cubic-Hermite long-range fade to zero
    dist >= max_dis          -> zero

with ``eps(d)`` the sigmoidal-dielectric function

    eps(d, D, D0, S) = D - 0.5*(D - D0)*(2 + 2*d*S + d^2*S^2)*exp(-d*S),

the electrostatic-units constant ``C1 = 322.0637`` and a baseline
shift ``C2 = C1 / (max_dis * eps(max_dis))`` so the Coulombic energy
goes to zero at the outer cutoff (before the fade).

Energies are multiplied by the count-pair connectivity weight
(:func:`tmol_jax.score.common.count_pair.connectivity_weight`):

    path_length >  4  -> 1.0
    path_length == 4  -> 0.2
    path_length <  4  -> 0.0

Virtual atoms (``e_i == 0`` or ``e_j == 0``) short-circuit to zero in
the upstream code, but here we let ``jnp.where`` handle it — the
multiplication by ``eiej`` naturally zeroes those pairs in every
region, so we just avoid Python branches for ``jit`` / ``vmap``
safety.
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.score.common.count_pair import connectivity_weight
from tmol_jax.score.common.cubic_hermite_polynomial import (
    interpolate,
    interpolate_to_zero,
)
from tmol_jax.types import Array

# The Rosetta electrostatic energy constant (kcal-Angstrom per e^2).
_C1: float = 322.0637


# --------------------------------------------------------------------- #
# Dielectric function and its derivative                                 #
# --------------------------------------------------------------------- #


def eps(dist: Array, D: Array, D0: Array, S: Array) -> Array:
    """Sigmoidal distance-dependent dielectric :math:`\\varepsilon(d)`.

    ``D`` is the large-distance asymptote, ``D0`` the small-distance
    limit, and ``S`` the inverse characteristic length of the switch.
    Defined on all ``dist >= 0``.
    """
    dS = dist * S
    return D - 0.5 * (D - D0) * (2.0 + 2.0 * dS + dS * dS) * jnp.exp(-dS)


def deps_ddist(dist: Array, D: Array, D0: Array, S: Array) -> Array:
    """Derivative ``d eps / d dist`` of the sigmoidal dielectric.

    Matches the closed-form from ``potentials.hh::deps_ddist``. Used
    when stitching the short- and long-range cubic fades at the
    seam with the Coulombic region.
    """
    return 0.5 * (D - D0) * dist * dist * (S * S * S) * jnp.exp(-dist * S)


# --------------------------------------------------------------------- #
# Pairwise elec energy                                                   #
# --------------------------------------------------------------------- #


def elec_pair(
    dist: Array,
    e_i: Array,
    e_j: Array,
    bonded_path_length: Array,
    global_params,
) -> Array:
    """Pairwise electrostatic energy for charges ``e_i`` / ``e_j``.

    Parameters
    ----------
    dist:
        Distance array (broadcastable with ``e_i`` / ``e_j`` /
        ``bonded_path_length``). Any units match those used to build
        ``global_params`` -- Angstrom in all standard tmol data files.
    e_i, e_j:
        Partial charges on the two atoms.
    bonded_path_length:
        Shortest chemical-path distance between the two atoms in # of
        bonds. Floats are accepted; the count-pair weight thresholds
        at 4 use integer-style comparisons that also work for floats.
    global_params:
        An :class:`~tmol_jax.score.elec.params.ElecGlobalParams` -- only
        the ``D``, ``D0``, ``S``, ``min_dis``, ``max_dis`` scalars are
        read.

    Returns
    -------
    Energy in kcal/mol, same shape as the inputs. Safe under ``jit``
    and ``vmap`` -- the four-region dispatch is expressed as a chain
    of :func:`jax.numpy.where` selectors, without any Python branching
    on traced values.

    Notes
    -----
    Upstream's C++ special-cases ``eiej == 0`` for an early return.
    The JAX code cannot branch on traced scalars; instead every region
    multiplies by ``eiej`` so virtual atoms (charge 0) contribute 0.
    """

    D = global_params.D
    D0 = global_params.D0
    S = global_params.S
    min_dis = global_params.min_dis
    max_dis = global_params.max_dis

    low_poly_start = min_dis - 0.25
    low_poly_end = min_dis + 0.25
    hi_poly_start = max_dis - 1.0
    hi_poly_end = max_dis

    # ``_C1`` is a plain Python float; don't force it to ``float32`` here
    # because a hard cast would truncate precision when the caller is
    # running at float64 (e.g. finite-difference gradchecks). Using the
    # raw Python float lets JAX promote it to match ``dist``'s dtype,
    # preserving whatever precision the caller asked for.
    C1 = _C1
    # Baseline shift: E(max_dis) = 0 (before the long-range fade kicks in).
    C2 = C1 / (max_dis * eps(max_dis, D, D0, S))

    eiej = e_i * e_j

    # ----- Region 1 & 2 seam: value/derivative of the Coulombic piece
    # at ``low_poly_end`` (seam between short-range fade and Coulomb).
    eps_low_end = eps(low_poly_end, D, D0, S)
    deps_low_end = deps_ddist(low_poly_end, D, D0, S)
    min_dis_score_unscaled = C1 / (min_dis * eps(min_dis, D, D0, S)) - C2
    flat_E = eiej * min_dis_score_unscaled  # Region 1 value

    dmax_elec = eiej * (C1 / (low_poly_end * eps_low_end) - C2)
    # Region 2: cubic Hermite on [low_poly_start, low_poly_end], matching
    # value/derivative (flat_E, 0) at the left and (dmax_elec, dmax_elec_ddist)
    # at the right. The right-end derivative is d/ddist of the *full Coulombic
    # energy* at low_poly_end (upstream potentials.hh:172-185) -- the same form
    # Region 4 already uses below -- NOT the bare dielectric slope.
    dmax_elec_ddist = (
        -C1
        * eiej
        * (eps_low_end + low_poly_end * deps_low_end)
        / (low_poly_end * low_poly_end * eps_low_end * eps_low_end)
    )
    short_fade_E = interpolate(
        dist,
        low_poly_start,
        flat_E,
        # Pass 0.0 as a Python float so JAX promotes it to match the
        # working precision (float32 in production, float64 under
        # ``jax_enable_x64`` for gradchecks); a hard ``jnp.float32`` cast
        # here would pin the intermediate to 32 bits and leak precision.
        0.0,
        low_poly_end,
        dmax_elec,
        dmax_elec_ddist,
    )

    # ----- Region 3: "true" Coulombic.
    # ``dist`` can be 0 on the non-matching branches of the where
    # cascade; safeguard against /0 there so autograd doesn't get NaNs
    # flowing back up.
    # Use a plain Python ``1.0`` so this path inherits ``dist``'s dtype
    # rather than forcing float32; see note on ``C1`` above.
    safe_dist = jnp.where(dist > 0.0, dist, 1.0)
    eps_d = eps(safe_dist, D, D0, S)
    coulombic_E = eiej * (C1 / (safe_dist * eps_d) - C2)

    # ----- Region 4: long-range cubic fade to zero.
    eps_hi_start = eps(hi_poly_start, D, D0, S)
    deps_hi_start = deps_ddist(hi_poly_start, D, D0, S)
    dmin_elec = eiej * (C1 / (hi_poly_start * eps_hi_start) - C2)
    dmin_elec_ddist = (
        -C1
        * eiej
        * (eps_hi_start + hi_poly_start * deps_hi_start)
        / (hi_poly_start * hi_poly_start * eps_hi_start * eps_hi_start)
    )
    long_fade_E = interpolate_to_zero(
        dist, hi_poly_start, dmin_elec, dmin_elec_ddist, hi_poly_end
    )

    # ----- Dispatch: choose the right region per element.
    #
    # Order matters: we walk from the far field in, so the inner-most
    # region wins. Region 5 (dist >= max_dis) is the "else" branch of
    # the initial zero.
    E = jnp.zeros_like(dist * eiej)
    E = jnp.where(dist < hi_poly_end, long_fade_E, E)
    E = jnp.where(dist < hi_poly_start, coulombic_E, E)
    E = jnp.where(dist < low_poly_end, short_fade_E, E)
    E = jnp.where(dist < low_poly_start, flat_E, E)

    # Count-pair weight. ``connectivity_weight`` already zeroes
    # 1-1/1-2/1-3 pairs, so path-length masking is built in.
    weight = connectivity_weight(bonded_path_length)
    return weight * E


__all__ = ["eps", "deps_ddist", "elec_pair"]
