"""Backbone-torsion scoring (rama + bb-dependent omega), pure JAX.

JAX port of :mod:`tmol.score.backbone_torsion`. The term computes two
score channels that both live on the residue ``i`` / ``(phi_i, psi_i,
omega_i)`` triple:

* :data:`ScoreType.rama` — bilinear lookup in a per-AA ``[n_phi, n_psi]``
  Ramachandran energy grid (kcal/mol).
* :data:`ScoreType.omega` — harmonic penalty around a bb-dependent
  ``omega0(phi, psi)`` with spring constant ``k(phi, psi)``, also
  obtained from a bilinear ``(phi, psi)`` lookup.

Public surface:

* :func:`~tmol_jax.score.backbone_torsion.rama.rama_score`
* :class:`~tmol_jax.score.backbone_torsion.rama.RamaDatabaseAccess`
* :func:`~tmol_jax.score.backbone_torsion.omega.omega_score`
* :class:`~tmol_jax.score.backbone_torsion.omega.OmegaBBDepDatabaseAccess`
* :class:`~tmol_jax.score.backbone_torsion.omega.OmegaParams`
* :class:`~tmol_jax.score.backbone_torsion.backbone_torsion_energy_term.BackboneTorsionEnergyTerm`

Internal kernels (bilinear interpolation, angle wrapping, the raw
``rama_potential`` / ``omega_potential`` functions) live in
:mod:`~tmol_jax.score.backbone_torsion.potentials` and are re-exported
here for callers that just want the numerics without the pose-stack
plumbing.
"""

from __future__ import annotations

from tmol_jax.score.backbone_torsion.backbone_torsion_energy_term import (
    BackboneTorsionEnergyTerm,
    BackboneTorsionTermState,
)
from tmol_jax.score.backbone_torsion.omega import (
    OmegaBBDepDatabaseAccess,
    OmegaParams,
    omega_score,
)
from tmol_jax.score.backbone_torsion.potentials import (
    bilinear_interpolate,
    omega_potential,
    rama_potential,
    wrap_angle,
)
from tmol_jax.score.backbone_torsion.rama import (
    RamaDatabaseAccess,
    rama_score,
)

__all__ = [
    "BackboneTorsionEnergyTerm",
    "BackboneTorsionTermState",
    "OmegaBBDepDatabaseAccess",
    "OmegaParams",
    "RamaDatabaseAccess",
    "bilinear_interpolate",
    "omega_potential",
    "omega_score",
    "rama_potential",
    "rama_score",
    "wrap_angle",
]
