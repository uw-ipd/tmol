"""Reference-energy score term.

Direct port of :mod:`tmol.score.ref` to pure JAX. The reference term adds
a single, conformation-independent, per-residue-type constant to the
total score. It exists solely to balance composition preferences in the
full score function (e.g. slightly penalizing Pro vs Ala).

Only the :class:`~tmol_jax.score.ref.ref_energy_term.RefEnergyTerm`
class is publicly exported; callers never instantiate it directly — they
register it via :meth:`ScoreFunction.set_weight` with
:data:`ScoreType.ref` and the scoring machinery does the rest.
"""

from tmol_jax.score.ref.ref_energy_term import RefEnergyTerm

__all__ = ["RefEnergyTerm"]
