"""Cartesian-bonded harmonic / periodic potentials (pure JAX).

Direct port of the scalar functions in
``tmol/score/cartbonded/potentials/potentials.hh``. Every function here is
vectorized over leading batch axes: the geometric inputs (``d``, ``theta``,
``phi``) and parameter inputs (``d0``, ``k``, ...) broadcast together and
the returned energy has the broadcast shape.

Derivatives are not written by hand -- :func:`jax.grad` produces them from
these value functions. The upstream ``_V_dV`` helpers bake the chain rule
through ``distance`` / ``pt_interior_angle`` / ``dihedral_angle`` into C++
templates; we rely on JAX autodiff to do the same.

Functions
---------
* :func:`bond_length_E` -- harmonic in distance.
* :func:`bond_angle_E`  -- harmonic in interior angle.
* :func:`torsion_E`     -- single-cosine periodic potential for a proper
  torsion, ``0.5 * k * (1 + cos(n*phi - phi0))``.
* :func:`improper_E`    -- harmonic in improper torsion angle.
* :func:`hxltorsion_E`  -- sum of three cosines for hydroxyl torsions,
  ``sum_n k_n * (1 + cos(n*phi - phi_n))`` (no 0.5 prefactor, matching the
  upstream ``cbhxltorsion_V_dV_old`` and ``cbtorsion_V_dV`` helpers).
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.types import Array


def bond_length_E(d: Array, d0: Array, k: Array) -> Array:
    """Harmonic bond-length penalty ``0.5 * k * (d - d0)^2``.

    Parameters
    ----------
    d
        Current bond length, any broadcastable shape.
    d0
        Ideal bond length, broadcastable with ``d``.
    k
        Spring constant, broadcastable with ``d``.

    Returns
    -------
    Array
        Energy, same broadcast shape as the inputs.
    """
    return 0.5 * k * jnp.square(d - d0)


def bond_angle_E(theta: Array, theta0: Array, k: Array) -> Array:
    """Harmonic bond-angle penalty ``0.5 * k * (theta - theta0)^2``.

    All inputs are in radians and broadcast together.
    """
    return 0.5 * k * jnp.square(theta - theta0)


def torsion_E(phi: Array, phi0: Array, k: Array, n: Array) -> Array:
    """Single-cosine periodic torsion potential.

    ``E = 0.5 * k * (1 + cos(n * phi - phi0))``.

    Parameters
    ----------
    phi
        Current torsion angle (radians).
    phi0
        Phase offset (radians).
    k
        Force constant.
    n
        Periodicity (integer or float, broadcast as a multiplier).

    Notes
    -----
    Use :func:`hxltorsion_E` for a sum of three cosines (the Rosetta
    ``cbtorsion`` / hydroxyl-torsion form); use :func:`improper_E` for
    improper torsions that are harmonic rather than periodic.
    """
    return 0.5 * k * (1.0 + jnp.cos(n * phi - phi0))


def improper_E(chi: Array, chi0: Array, k: Array) -> Array:
    """Harmonic improper-torsion penalty ``0.5 * k * (chi - chi0)^2``.

    All inputs are in radians and broadcast together.
    """
    return 0.5 * k * jnp.square(chi - chi0)


def hxltorsion_E(
    phi: Array,
    k1: Array,
    k2: Array,
    k3: Array,
    phi1: Array,
    phi2: Array,
    phi3: Array,
) -> Array:
    """Hydroxyl-torsion sum-of-three-cosines potential.

    ``E = sum_n k_n * (1 + cos(n * phi - phi_n))`` for ``n = 1, 2, 3``.

    Matches ``cbhxltorsion_V_dV_old`` and the ``cbtorsion`` helper in
    ``potentials.hh``; note there is no 0.5 prefactor on the terms, in
    contrast to :func:`torsion_E`.
    """
    return (
        k1 * (1.0 + jnp.cos(1.0 * phi - phi1))
        + k2 * (1.0 + jnp.cos(2.0 * phi - phi2))
        + k3 * (1.0 + jnp.cos(3.0 * phi - phi3))
    )


__all__ = [
    "bond_length_E",
    "bond_angle_E",
    "torsion_E",
    "improper_E",
    "hxltorsion_E",
]
