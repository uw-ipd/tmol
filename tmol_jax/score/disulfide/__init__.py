"""Disulfide score term (pure JAX).

JAX port of :mod:`tmol.score.disulfide`. Evaluates a Rosetta
beta_nov2016-style disulfide potential on Cys-Cys pairs whose sulfur
atoms lie within 3.5 Å: a weighted sum of an SG-SG distance component,
two CA-CB-S angle components, a CB-S-S-CB dihedral mixture, and two
CA-CB-S-S dihedral mixtures.

Public surface:

* :func:`~tmol_jax.score.disulfide.potentials.disulfide_pair_E`
  — pair energy for a single Cys-Cys pair.
* :class:`~tmol_jax.score.disulfide.disulfide_energy_term.DisulfideEnergyTerm`
  — whole-pose scoring term that provides
  :data:`~tmol_jax.score.score_types.ScoreType.disulfide`.
"""

from __future__ import annotations

from tmol_jax.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol_jax.score.disulfide.potentials import disulfide_pair_E

__all__ = [
    "DisulfideEnergyTerm",
    "disulfide_pair_E",
]
