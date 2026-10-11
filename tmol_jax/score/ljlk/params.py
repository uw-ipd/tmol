"""LJ / LK parameter containers.

Upstream reference: :mod:`tmol.score.ljlk.params` and the C++
``LJTypeParams`` / ``LJLKTypeParams`` / ``LJGlobalParams`` in
``tmol/score/ljlk/potentials/params.hh``.

The C++ side defines three overlapping structs (``LJTypeParams``,
``LKTypeParams``, ``LJLKTypeParams``) that all carry the five numeric
parameters plus four H-bond flags. The JAX port merges them into a
single :class:`LJTypeParams` dataclass — JAX broadcasts cleanly so the
split into "LJ" / "LK" sub-params adds no value here. The container is
registered as a pytree so it flows through ``jit``/``vmap``.

Loading:

    from tmol_jax.database.scoring.ljlk import LJLKDatabase
    from tmol_jax.database.chemical import ChemicalDatabase
    db = LJLKDatabase.from_file("ljlk.yaml")
    chem = ChemicalDatabase.get_default()
    type_params = LJTypeParams.from_database(db, chem)
    global_params = LJLKGlobal.from_database(db)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import as_jnp, pytree_dataclass

# --------------------------------------------------------------------------- #
# Containers                                                                   #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class LJTypeParams:
    """Per-atom-type LJ + LK parameters, indexed by atom-type index.

    Shapes: every field is a 1-D JAX array of length ``n_atom_types``.
    The boolean fields are stored as ``bool_`` arrays rather than floats
    so ``jnp.where`` works naturally; they broadcast against float
    arrays exactly as expected.

    Fields
    ------
    lj_radius
        Atom radius (A). Used to form ``sigma`` via ``lj_sigma``.
    lj_wdepth
        Well-depth (kcal/mol). ``epsilon`` is the geometric mean over a
        pair ``sqrt(w_i * w_j)``.
    is_donor
        True for atoms acting as H-bond donors (heavy-atom side).
    is_acceptor
        True for atoms acting as H-bond acceptors.
    is_hydroxyl
        True for hydroxyl donors; selects
        ``global.lj_hbond_OH_donor_dis``.
    is_polarh
        True for polar hydrogens.
    lk_volume
        LK solvent-accessible volume.
    lk_dgfree
        LK desolvation free energy (kcal/mol).
    lk_lambda
        LK Gaussian width parameter.
    """

    lj_radius: jax.Array
    lj_wdepth: jax.Array
    is_donor: jax.Array
    is_acceptor: jax.Array
    is_hydroxyl: jax.Array
    is_polarh: jax.Array
    is_carbon_lk: jax.Array
    lk_volume: jax.Array
    lk_dgfree: jax.Array
    lk_lambda: jax.Array

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_database(cls, ljlk_db: Any, chem_db: Any) -> "LJTypeParams":
        """Build packed per-atom-type parameters indexed against
        ``chem_db.atom_types``.

        Every atom type in the chemical database is resolved against
        ``ljlk_db.atom_type_parameters``. Types missing from the LJLK
        table (shouldn't happen for the canonical beta_nov2016 palette,
        but guards against custom atoms) fall back to zero-valued LJ/LK
        parameters, with flags copied from the chem-db entry.
        """
        lj_by_name = {p.name: p for p in ljlk_db.atom_type_parameters}
        n = len(chem_db.atom_types)

        # Use safely-positive defaults to keep gradients finite through
        # padded/virtual atoms. The downstream real-atom mask zeros the
        # contribution out of the sum, but intermediate values must be
        # finite so that reverse-mode autograd doesn't propagate NaN.
        lj_radius = np.full(n, 1.7, dtype=np.float32)
        lj_wdepth = np.zeros(n, dtype=np.float32)
        lk_volume = np.zeros(n, dtype=np.float32)
        lk_dgfree = np.zeros(n, dtype=np.float32)
        lk_lambda = np.ones(n, dtype=np.float32)  # avoid divide-by-zero
        is_donor = np.zeros(n, dtype=np.bool_)
        is_acceptor = np.zeros(n, dtype=np.bool_)
        is_hydroxyl = np.zeros(n, dtype=np.bool_)
        is_polarh = np.zeros(n, dtype=np.bool_)

        for i, at in enumerate(chem_db.atom_types):
            is_donor[i] = bool(at.is_donor)
            is_acceptor[i] = bool(at.is_acceptor)
            is_hydroxyl[i] = bool(at.is_hydroxyl)
            is_polarh[i] = bool(at.is_polarh)
            lj = lj_by_name.get(at.name)
            if lj is not None:
                lj_radius[i] = lj.lj_radius
                lj_wdepth[i] = lj.lj_wdepth
                lk_volume[i] = lj.lk_volume
                lk_dgfree[i] = lj.lk_dgfree
                lk_lambda[i] = lj.lk_lambda

        return cls(
            lj_radius=jnp.asarray(lj_radius),
            lj_wdepth=jnp.asarray(lj_wdepth),
            is_donor=jnp.asarray(is_donor),
            is_acceptor=jnp.asarray(is_acceptor),
            is_hydroxyl=jnp.asarray(is_hydroxyl),
            is_polarh=jnp.asarray(is_polarh),
            is_carbon_lk=jnp.asarray(
                [at.name in {"CH1", "CH2", "CH3", "Caro"} for at in chem_db.atom_types]
            ),
            lk_volume=jnp.asarray(lk_volume),
            lk_dgfree=jnp.asarray(lk_dgfree),
            lk_lambda=jnp.asarray(lk_lambda),
        )

    # ------------------------------------------------------------------ #
    # Gather                                                             #
    # ------------------------------------------------------------------ #
    def gather(self, idx: jax.Array) -> "LJTypeParams":
        """Gather per-atom-type parameters along the 0-th axis.

        Returns a new :class:`LJTypeParams` where every field has been
        indexed with ``idx``. Out-of-range indices (``idx < 0``) are
        clamped to 0; callers should mask the result with
        ``idx >= 0`` if they care about padding.
        """
        safe = jnp.where(idx >= 0, idx, 0)
        return LJTypeParams(
            lj_radius=self.lj_radius[safe],
            lj_wdepth=self.lj_wdepth[safe],
            is_donor=self.is_donor[safe],
            is_acceptor=self.is_acceptor[safe],
            is_hydroxyl=self.is_hydroxyl[safe],
            is_polarh=self.is_polarh[safe],
            is_carbon_lk=self.is_carbon_lk[safe],
            lk_volume=self.lk_volume[safe],
            lk_dgfree=self.lk_dgfree[safe],
            lk_lambda=self.lk_lambda[safe],
        )


@pytree_dataclass
class LJLKGlobal:
    """LJ / LK global parameters (one scalar each).

    ``max_dis`` is the far cpoly cutoff (``cpoly_dmax``) and
    ``lj_dlin_sigma_factor`` is the multiplier in ``d_lin = sigma *
    factor`` that sets the linear-extrapolation crossover on the LJ
    repulsive wall. Both are threaded through to the potentials so a
    non-default YAML override changes behaviour without code edits.
    The three H-bond sigma overrides select per-pair ``sigma`` in
    :func:`tmol_jax.score.ljlk.potentials.lj_sigma`.
    """

    lj_hbond_dis: jax.Array
    lj_hbond_OH_donor_dis: jax.Array
    lj_hbond_hdis: jax.Array
    max_dis: jax.Array
    lj_dlin_sigma_factor: jax.Array

    @classmethod
    def from_database(cls, ljlk_db: Any) -> "LJLKGlobal":
        gp = ljlk_db.global_parameters
        return cls(
            lj_hbond_dis=as_jnp(gp.lj_hbond_dis, dtype=jnp.float32),
            lj_hbond_OH_donor_dis=as_jnp(gp.lj_hbond_OH_donor_dis, dtype=jnp.float32),
            lj_hbond_hdis=as_jnp(gp.lj_hbond_hdis, dtype=jnp.float32),
            max_dis=as_jnp(gp.max_dis, dtype=jnp.float32),
            lj_dlin_sigma_factor=as_jnp(gp.lj_dlin_sigma_factor, dtype=jnp.float32),
        )


__all__ = ["LJTypeParams", "LJLKGlobal"]
