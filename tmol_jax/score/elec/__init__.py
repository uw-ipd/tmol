"""Electrostatics score term (pure JAX).

JAX port of :mod:`tmol.score.elec`. Implements the Rosetta-style
sigmoidal-dielectric Coulomb interaction with short- and long-range
cubic-Hermite fades, weighted by the count-pair connectivity factor.

Public surface:

* :class:`~tmol_jax.score.elec.params.ElecGlobalParams` — the five
  sigmoidal-dielectric / cutoff constants, as JAX scalars.
* :func:`~tmol_jax.score.elec.params.atom_charges_for_pose_stack` —
  looks up partial charges for every atom in a PoseStack.
* :func:`~tmol_jax.score.elec.potentials.eps` — sigmoidal-dielectric
  function :math:`\\varepsilon(d)`.
* :func:`~tmol_jax.score.elec.potentials.elec_pair` — pairwise
  Coulombic energy for a charge pair at a given distance and chemical
  path length.
* :class:`~tmol_jax.score.elec.elec_energy_term.ElecEnergyTerm` — the
  full whole-pose scoring term (``ScoreType.fa_elec``).
"""

from __future__ import annotations

from tmol_jax.score.elec.params import (
    ElecGlobalParams,
    atom_charges_for_pose_stack,
    elec_global_params_from_database,
)
from tmol_jax.score.elec.potentials import eps, elec_pair
from tmol_jax.score.elec.elec_energy_term import ElecEnergyTerm

__all__ = [
    "ElecGlobalParams",
    "atom_charges_for_pose_stack",
    "elec_global_params_from_database",
    "eps",
    "elec_pair",
    "ElecEnergyTerm",
]
