"""Pure-JAX disulfide pair potential.

The upstream tmol disulfide term (``tmol/score/disulfide/potentials/``)
evaluates, for every Cys-Cys pair in chemical-bonding range, a weighted
sum of four 1-D potentials:

1. **S-S distance** — a skew-normal Weibull-like log-density of
   ``ssdist = |SG_i - SG_j|``.
2. **CA-CB-S angle, each cysteine** — a von Mises at ``angle(CB, SG, SG')``.
3. **CB-S-S-CB dihedral** — a mixture of two von Mises peaks at the
   ``dihedral(CB_i, SG_i, SG_j, CB_j)`` angle.
4. **CA-CB-S-S dihedral, each cysteine** — a mixture of three von Mises
   peaks at the ``dihedral(CA, CB, SG, SG')`` angle.

All components and their weights come from the small global-parameter
struct baked into :class:`tmol_jax.database.scoring.disulfide.DisulfideDatabase`
via the bundled ``disulfide.yaml``. The function is ``jit``-safe,
``vmap``-friendly, and has no Python-level control flow that depends on
the (traced) coordinates.

The potentials are implemented with stable numerics: all mixture
log-sum-exp steps add a ``MEST = exp(-20)`` floor before the outer
log, matching the upstream C++ reference.

Although the user-facing spec describes each component as "a polynomial",
the Rosetta-derived fits in ``disulfide.yaml`` are Weibull / von Mises
densities; we expose :func:`_polyval` as a thin re-export of
``jnp.polyval`` for callers that wish to evaluate straight polynomials
in the same idiom.
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.score.common.geom import (
    dihedral_angle,
    distance,
    pt_interior_angle,
)
from tmol_jax.types import Array

# ``exp(-20)`` floor inside the outer ``log`` of each mixture-of-peaks
# term. Matches the upstream C++ implementation (``MEST = exp(-20.0)``).
_MEST = jnp.exp(jnp.asarray(-20.0))


def _polyval(coeffs: Array, x: Array) -> Array:
    """Evaluate a standard polynomial via Horner's method.

    Re-exports :func:`jax.numpy.polyval` for term-local parity with
    the upstream polynomial idiom (coefficients in descending-power
    order). The disulfide potentials use Weibull / von Mises forms, not
    plain polynomials, but some callers still want a ``jnp.polyval``
    shortcut keyed by this module.
    """
    return jnp.polyval(coeffs, x)


def _distance_score(ssdist: Array, p) -> Array:
    """Skew-normal (Weibull-flavoured) log-density at ``ssdist``."""
    z = (ssdist - p["d_location"]) / p["d_scale"]
    # ``erfc(-a z / sqrt(2))`` is the skewness factor.
    erfc_arg = -p["d_shape"] * z / jnp.sqrt(2.0)
    return 0.5 * z * z - jnp.log(jax_erfc(erfc_arg) + _MEST)


def jax_erfc(x: Array) -> Array:
    """Accurate ``erfc(x)`` via :func:`jax.scipy.special.erfc`.

    Was ``1 - erf(x)``, which catastrophically cancels for large ``x``
    (erf rounds to >=1 -> erfc 0 or slightly negative). That both (a)
    broke ``_distance_score``'s ``log(erfc + MEST)`` under jit (erfc+MEST
    collapsed to 0 -> log=-inf -> +inf score on strained disulfides) and
    (b) gave slightly wrong values. ``jax.scipy.special.erfc`` is accurate
    and always >= 0, matching upstream's ``std::erfc``.
    """
    from jax.scipy.special import erfc

    return erfc(x)


def _vm1_score(angle: Array, logA: Array, kappa: Array, mu: Array) -> Array:
    """Negative log of a single von Mises density (up to normalisation)."""
    return -logA - kappa * jnp.cos(angle - mu)


def _vm_mixture_score(angle: Array, logAs, kappas, mus) -> Array:
    """-log of a sum-of-von-Mises mixture with a numerical floor.

    ``logAs``, ``kappas``, ``mus`` are tuples of arrays (one per peak).
    """
    exps = [
        jnp.exp(lA) * jnp.exp(k * jnp.cos(angle - m))
        for lA, k, m in zip(logAs, kappas, mus)
    ]
    total = exps[0]
    for e in exps[1:]:
        total = total + e
    return -jnp.log(total + _MEST)


def disulfide_pair_E(
    ca_i: Array,
    cb_i: Array,
    sg_i: Array,
    ca_j: Array,
    cb_j: Array,
    sg_j: Array,
    params,
) -> Array:
    """Energy of one Cys-Cys pair.

    Parameters
    ----------
    ca_i, cb_i, sg_i, ca_j, cb_j, sg_j:
        ``[..., 3]`` coordinates of the two cysteines' CA / CB / SG atoms.
        All six must share a common broadcast shape; the pair energy is
        computed element-wise over the leading batch dimensions.
    params:
        Mapping with the 26 disulfide global parameters (see
        :class:`tmol_jax.database.scoring.disulfide.DisulfideGlobalParameters`).
        Accepts either a ``dict`` or an object with attribute access.

    Returns
    -------
    ``float32[...]`` — the weighted sum of the four 1-D potentials, minus
    the global ``shift`` constant. Broadcasts over the leading batch
    dimensions of the coordinate inputs.
    """
    p = _params_as_dict(params)

    ssdist = distance(sg_i, sg_j)
    ang_i = pt_interior_angle(cb_i, sg_i, sg_j)
    ang_j = pt_interior_angle(cb_j, sg_j, sg_i)
    dihed_ss = dihedral_angle(cb_i, sg_i, sg_j, cb_j)
    dihed_cs_i = dihedral_angle(ca_i, cb_i, sg_i, sg_j)
    dihed_cs_j = dihedral_angle(ca_j, cb_j, sg_j, sg_i)

    e = -p["shift"]

    # Distance.
    e = e + p["wt_len"] * _distance_score(ssdist, p)

    # CA-CB-S angles.
    e = e + p["wt_ang"] * _vm1_score(ang_i, p["a_logA"], p["a_kappa"], p["a_mu"])
    e = e + p["wt_ang"] * _vm1_score(ang_j, p["a_logA"], p["a_kappa"], p["a_mu"])

    # CB-S-S-CB dihedral (2-peak mixture).
    e = e + p["wt_dih_ss"] * _vm_mixture_score(
        dihed_ss,
        (p["dss_logA1"], p["dss_logA2"]),
        (p["dss_kappa1"], p["dss_kappa2"]),
        (p["dss_mu1"], p["dss_mu2"]),
    )

    # CA-CB-S-S dihedrals (3-peak mixture, once per cysteine).
    logAs = (p["dcs_logA1"], p["dcs_logA2"], p["dcs_logA3"])
    kappas = (p["dcs_kappa1"], p["dcs_kappa2"], p["dcs_kappa3"])
    mus = (p["dcs_mu1"], p["dcs_mu2"], p["dcs_mu3"])
    e = e + p["wt_dih_cs"] * _vm_mixture_score(dihed_cs_i, logAs, kappas, mus)
    e = e + p["wt_dih_cs"] * _vm_mixture_score(dihed_cs_j, logAs, kappas, mus)

    return e


def _params_as_dict(params) -> dict:
    """Accept either a mapping or an attrs-style struct."""
    if isinstance(params, dict):
        return params
    # attrs / plain object — read attributes lazily on first use.
    names = (
        "d_location",
        "d_scale",
        "d_shape",
        "a_logA",
        "a_kappa",
        "a_mu",
        "dss_logA1",
        "dss_kappa1",
        "dss_mu1",
        "dss_logA2",
        "dss_kappa2",
        "dss_mu2",
        "dcs_logA1",
        "dcs_mu1",
        "dcs_kappa1",
        "dcs_logA2",
        "dcs_mu2",
        "dcs_kappa2",
        "dcs_logA3",
        "dcs_mu3",
        "dcs_kappa3",
        "wt_dih_ss",
        "wt_dih_cs",
        "wt_ang",
        "wt_len",
        "shift",
    )
    return {n: jnp.asarray(getattr(params, n)) for n in names}


__all__ = ["disulfide_pair_E"]
