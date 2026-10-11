"""LJ / LK (Lennard-Jones + Lazaridis-Karplus) energy term.

Pure-JAX port of ``tmol/score/ljlk``. The top-level C++ references this
port is faithful to are:

* ``tmol/score/ljlk/potentials/lj.hh`` — shifted + faded Lennard-Jones
  split into an attractive / repulsive pair.
* ``tmol/score/ljlk/potentials/lk_isotropic.hh`` — isotropic Lazaridis-
  Karplus desolvation, symmetrised over (i->j, j->i).
* ``tmol/score/ljlk/potentials/common.hh`` — shared ``lj_sigma`` and
  ``connectivity_weight`` helpers.

This package exposes three things:

* :class:`LJTypeParams`, :class:`LJLKGlobal` in :mod:`.params` -- flat
  attrs/pytree containers of the per-atom-type and global parameters.
* Pure-functional potentials in :mod:`.potentials` (``lj_sigma``,
  ``vdw``, ``vdw_and_deriv``, ``lj_atr_rep``, ``lk_isotropic_pair``).
* :class:`LJLKEnergyTerm` in :mod:`.ljlk_energy_term` -- an
  :class:`~tmol_jax.score.energy_term.EnergyTerm` subclass that
  produces ``{fa_ljatr, fa_ljrep, fa_lk}`` per-pose energies.

Everything is ``jit``-/``grad``-friendly: branches are resolved by
``jnp.where`` and distance cutoffs drop out of masked sums instead of
Python ``if`` statements.
"""

from __future__ import annotations

from tmol_jax.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol_jax.score.ljlk.params import LJLKGlobal, LJTypeParams
from tmol_jax.score.ljlk.potentials import (
    lj_atr_rep,
    lj_sigma,
    lk_isotropic_pair,
    vdw,
    vdw_and_deriv,
)

__all__ = [
    "LJLKEnergyTerm",
    "LJLKGlobal",
    "LJTypeParams",
    "lj_atr_rep",
    "lj_sigma",
    "lk_isotropic_pair",
    "vdw",
    "vdw_and_deriv",
]
