"""Vectorised hbond potentials in pure JAX.

Port of ``tmol/score/hbond/potentials/potentials.hh``. Each function
consumes 3-D coordinate arrays (leading batch axes preserved) plus a
*packed* polynomial-spec vector of length 15
``[c_a..c_k, x_lo, x_hi, y_lo, y_hi]`` — see :mod:`params` for the
convention and :class:`HBondPolynomials` for the full table.

The top-level entry point is :func:`hbond_pair_V`, which takes five
atomic positions plus all pair parameters and returns a scalar energy.
All helpers are pure and ``jit`` / ``vmap`` / ``grad`` friendly.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jss

from tmol_jax.score.common import (
    cos_interior_angle,
    dihedral_angle,
    distance,
    pt_interior_angle,
)
from tmol_jax.score.hbond.params import (
    HYBRIDIZATION_RING,
    HYBRIDIZATION_SP2,
)

# --------------------------------------------------------------------- #
# Packed poly helpers                                                    #
# --------------------------------------------------------------------- #


def _eval_packed_poly(x, poly):
    """Evaluate the bounded Chebyshev form produced by _poly_to_array."""
    lower, upper = poly[..., 11], poly[..., 12]
    valid = upper > lower
    width = jnp.where(valid, upper - lower, 1)
    # Clip the inactive branch to keep both values and derivatives finite.
    z = (
        2 * jnp.where(x < lower, lower, jnp.where(x > upper, upper, x)) - lower - upper
    ) / width
    b1 = jnp.zeros_like(z)
    b2 = jnp.zeros_like(z)
    for i in range(10, 0, -1):
        b1, b2 = 2 * z * b1 - b2 + poly[..., i], b1
    value = z * b1 - b2 + poly[..., 0]
    value = jnp.where(x < lower, poly[..., 13], value)
    value = jnp.where(x > upper, poly[..., 14], value)
    return jnp.where(valid, value, 0)


# --------------------------------------------------------------------- #
# Individual components                                                  #
# --------------------------------------------------------------------- #


def AH_dist_V(A, H, poly):
    """A-H distance polynomial. ``A, H: [..., 3]``, ``poly: [..., 15]``."""
    d = distance(A, H)
    return _eval_packed_poly(d, poly)


def AHD_angle_V(A, H, D, poly):
    """AHD angle polynomial (in angle space, not cos, matching the
    ``pt_interior_angle`` usage in ``potentials.hh``)."""
    ang = pt_interior_angle(A, H, D)
    return _eval_packed_poly(ang, poly)


def _BAH_base(B, A, H, poly):
    """cos(BAH) polynomial evaluated at the cosine of the interior
    angle between the vectors ``AH`` and ``AB``. Used as the core of
    the sp2 / ring / sp3 BAH forms."""
    AH = H - A
    BA = A - B
    cos_t = cos_interior_angle(AH, BA)
    return _eval_packed_poly(cos_t, poly)


def BAH_angle_V(B, B0, A, H, hybridization, poly, sp3_fade):
    """Hybridization-aware BAH angle term.

    * ``sp2``: evaluate against base atom ``B``.
    * ``ring``: evaluate against midpoint of B and B0.
    * ``sp3``: log-sum-exp of evaluations against B and B0 with
      temperature ``sp3_fade`` (soft-max fade in the upstream).

    The branching is done with ``jnp.where``, so all three branches are
    computed for every element but only the one matching
    ``hybridization`` contributes. This keeps the function ``jit`` /
    ``vmap`` friendly — hybridization can be a traced array.
    """
    # sp2 branch
    v_sp2 = _BAH_base(B, A, H, poly)

    # ring branch (evaluate with midpoint base)
    Bm = 0.5 * (B + B0)
    v_ring = _BAH_base(Bm, A, H, poly)

    # sp3 branch (soft-max fade between evaluations against B and B0)
    v_b = _BAH_base(B, A, H, poly)
    v_b0 = _BAH_base(B0, A, H, poly)
    # log( exp(a*f) + exp(b*f) ) / f  == logsumexp along stacked axis
    stacked = jnp.stack([v_b * sp3_fade, v_b0 * sp3_fade], axis=-1)
    v_sp3 = jss.logsumexp(stacked, axis=-1) / sp3_fade

    val = jnp.where(hybridization == HYBRIDIZATION_SP2, v_sp2, v_sp3)
    val = jnp.where(hybridization == HYBRIDIZATION_RING, v_ring, val)
    return val


def _sp2chi_energy(ang, chi, d, m, l):
    """Analytic sp2 chi-angle penalty (Rosetta sp2_elec).

    Parameters
    ----------
    ang:
        B-A-H angle (radians).
    chi:
        B0-B-A-H dihedral (radians).
    d:
        ``hb_sp2_BAH180_rise``.
    m:
        ``hb_sp2_range_span``.
    l:
        ``hb_sp2_outer_width``.

    See :func:`sp2chi_energy_V` in the C++ reference — this is a
    branch-free ``jnp.where`` rewrite.
    """
    pi = jnp.pi
    H = 0.5 * (jnp.cos(2.0 * chi) + 1.0)

    # Region 1: ang > 2*pi/3
    F1 = 0.5 * d * jnp.cos(3.0 * (pi - ang)) + 0.5 * d - 0.5
    G1 = d - 0.5
    E1 = H * F1 + (1.0 - H) * G1

    # Region 2: pi*(2/3 - l) <= ang <= 2*pi/3. The outer-rise cosine
    # expects ``(2*pi/3 - ang) / l`` to span [0, pi] across this
    # interval, i.e. the region width is ``pi*l`` not ``l`` — matching
    # the upstream boundary ``pi * (2.0 / 3.0 - l)`` in
    # ``potentials.hh::sp2chi_energy_V``.
    outer_rise = jnp.cos(pi - (2.0 * pi / 3.0 - ang) / l)
    F2 = 0.5 * m * outer_rise + 0.5 * m - 0.5
    G2 = 0.5 * (m - d) * outer_rise + 0.5 * (m - d) + d - 0.5
    E2 = H * F2 + (1.0 - H) * G2

    # Region 3 (default): ang < pi*(2/3 - l). Upstream boundary is
    # ``pi * (2.0 / 3.0 - l)`` — note the leading factor of pi.
    E3 = m - 0.5

    E = jnp.where(ang > 2.0 * pi / 3.0, E1, E2)
    E = jnp.where(ang < pi * (2.0 / 3.0 - l), E3, E)
    return E


def chi_angle_V(D, H, A, B, B0, hybridization, global_params):
    """sp2 chi penalty on the BAH angle + B0-B-A-H dihedral.

    Returns 0 for non-sp2 acceptors (matches the C++ behaviour — sp3
    and ring hybridizations have no chi term).
    """
    BAH = pt_interior_angle(B, A, H)
    B0BAH = dihedral_angle(B0, B, A, H)
    E_sp2 = _sp2chi_energy(
        BAH,
        B0BAH,
        global_params.hb_sp2_BAH180_rise,
        global_params.hb_sp2_range_span,
        global_params.hb_sp2_outer_width,
    )
    return jnp.where(hybridization == HYBRIDIZATION_SP2, E_sp2, jnp.zeros_like(E_sp2))


# --------------------------------------------------------------------- #
# Fade near zero                                                         #
# --------------------------------------------------------------------- #


def _fade_near_zero(E):
    """Piecewise smooth truncation (matches ``potentials.hh`` logic).

    * ``E >  0.1``       → ``0``  (weak interactions zeroed)
    * ``-0.1 <= E <=  0.1`` → smooth cubic fade
      ``-0.025 + 0.5 E - 2.5 E^2``
    * ``E < -0.1``       → ``E`` (unchanged)
    """
    faded = -0.025 + 0.5 * E - 2.5 * E * E
    E1 = jnp.where(E > 0.1, jnp.zeros_like(E), E)
    E2 = jnp.where(E > -0.1, faded, E1)
    # And the "E > 0.1" clause still wins.
    return jnp.where(E > 0.1, jnp.zeros_like(E), E2)


# --------------------------------------------------------------------- #
# Top-level pair energy                                                  #
# --------------------------------------------------------------------- #


def hbond_pair_V(
    D,
    H,
    A,
    B,
    B0,
    hybridization,
    pair_weight,
    AHdist_poly,
    cosAHD_poly,
    cosBAH_poly,
    global_params,
):
    """Full hbond pair energy.

    ``D, H, A, B, B0`` are ``[..., 3]`` coordinate arrays. The remaining
    arguments are scalar / ``[..., 15]`` polynomial specs paired with
    the (donor_type, acceptor_type) of the leading batch index.

    Returns a scalar (or batched scalar) energy.

    The pipeline: sum of four pre-weight components (AH distance,
    AHD angle, BAH angle, sp2 chi), then multiply by the pair weight,
    then apply the near-zero fade.
    """
    e_ah = AH_dist_V(A, H, AHdist_poly)
    e_ahd = AHD_angle_V(A, H, D, cosAHD_poly)
    e_bah = BAH_angle_V(
        B,
        B0,
        A,
        H,
        hybridization,
        cosBAH_poly,
        global_params.hb_sp3_softmax_fade,
    )
    e_chi = chi_angle_V(D, H, A, B, B0, hybridization, global_params)

    E = e_ah + e_ahd + e_bah + e_chi
    E = E * pair_weight
    return _fade_near_zero(E)


__all__ = [
    "AH_dist_V",
    "AHD_angle_V",
    "BAH_angle_V",
    "_BAH_base",
    "_sp2chi_energy",
    "chi_angle_V",
    "hbond_pair_V",
]
