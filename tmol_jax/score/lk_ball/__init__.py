"""LK-ball (anisotropic Lazaridis-Karplus) scoring in pure JAX.

Upstream reference: :mod:`tmol.score.lk_ball`. LK-ball extends the
isotropic Lazaridis-Karplus desolvation term to be *directional*: each
polar atom is given a small set of "water sites" along its preferred
hydrogen-bonding geometry, and the desolvation penalty between two
atoms is scaled up when an occluding atom is placed near one of those
waters.

Four score types are exported:

* ``lk_ball_iso``    — same as the isotropic LK term (not water-aware).
* ``lk_ball``        — ``lk_iso * desolv_fraction`` (anisotropic).
* ``lk_bridge``      — ``lk_iso * water_overlap_fraction`` (water-mediated).
* ``lk_bridge_uncpl``— ``water_overlap_fraction / 2``
                       (same geometry, not scaled by LK magnitude).

The port matches the upstream algorithm where feasible; see the
docstrings in :mod:`.potentials` and :mod:`.lk_ball_energy_term`
for the full list of simplifications (notably: an optional
"parity-preserving" energy-term path that returns zeros for all four
channels when the upstream infrastructure is not yet available).
"""

from __future__ import annotations

from tmol_jax.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol_jax.score.lk_ball.potentials import (
    LK_BALL_DEFAULTS,
    f_desolv,
    lk_ball_pair,
    lk_bridge_fraction,
    lk_fraction,
)

__all__ = [
    "LKBallEnergyTerm",
    "LK_BALL_DEFAULTS",
    "f_desolv",
    "lk_ball_pair",
    "lk_bridge_fraction",
    "lk_fraction",
]
