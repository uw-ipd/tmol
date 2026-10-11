"""Dunbrack rotamer-library energy term for tmol_jax.

Pure-JAX port of ``tmol.score.dunbrack``. Implements the three Dunbrack
channels (``rot``, ``rotdev``, ``semirot``) over a :class:`PoseStack`
with graceful degradation when the binary rotamer-library tables are
unavailable (see :mod:`tmol_jax.database.scoring.dunbrack_libraries`).

Public surface
--------------
* :class:`DunbrackEnergyTerm` -- the energy-term entry point.
* :func:`dunbrack_rot_score`, :func:`dunbrack_rotdev_score`,
  :func:`dunbrack_semirot_score` -- low-level scoring primitives,
  importable on their own for testing / stand-alone use.
"""

from tmol_jax.score.dunbrack.dunbrack_energy_term import (
    DunbrackEnergyTerm,
    DunbrackTermState,
)
from tmol_jax.score.dunbrack.potentials import (
    classify_rotamer_bin,
    dunbrack_rot_score,
    dunbrack_rotdev_score,
    dunbrack_semirot_score,
)

__all__ = [
    "DunbrackEnergyTerm",
    "DunbrackTermState",
    "classify_rotamer_bin",
    "dunbrack_rot_score",
    "dunbrack_rotdev_score",
    "dunbrack_semirot_score",
]
