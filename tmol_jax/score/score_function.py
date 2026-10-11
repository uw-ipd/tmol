"""Weighted energy terms and explicitly rendered, reusable JAX scorers."""

from __future__ import annotations

import copy
import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from .score_types import ScoreType

_TERM_TYPES = {
    "ljlk": ("LJLKEnergyTerm", ("fa_ljatr", "fa_ljrep", "fa_lk")),
    "elec": ("ElecEnergyTerm", ("fa_elec",)),
    "hbond": ("HBondEnergyTerm", ("hbond",)),
    "cartbonded": (
        "CartBondedEnergyTerm",
        (
            "cart_lengths",
            "cart_angles",
            "cart_torsions",
            "cart_impropers",
            "cart_hxltorsions",
        ),
    ),
    "constraint": ("ConstraintEnergyTerm", ("constraint",)),
    "disulfide": ("DisulfideEnergyTerm", ("disulfide",)),
    "backbone_torsion": ("BackboneTorsionEnergyTerm", ("omega", "rama")),
    "dunbrack": (
        "DunbrackEnergyTerm",
        (
            "dunbrack_rot",
            "dunbrack_rotdev",
            "dunbrack_semirot",
        ),
    ),
    "lk_ball": (
        "LKBallEnergyTerm",
        (
            "lk_ball_iso",
            "lk_ball",
            "lk_bridge",
            "lk_bridge_uncpl",
        ),
    ),
    "ref": ("RefEnergyTerm", ("ref",)),
    "genbonded": ("GenBondedEnergyTerm", ("gen_torsions",)),
}
_TERM_FOR_TYPE = {
    ScoreType[name]: (module, class_name)
    for module, (class_name, names) in _TERM_TYPES.items()
    for name in names
}


class ScoreFunction:
    """Mutable score configuration; render once for repeated coordinate scoring.

    Rendering snapshots weights and term options. Changing this configuration
    requires rendering again; an existing scorer keeps its original behavior.
    Topology and atom layout are fixed for each rendered scorer. Coordinates
    are its only input, so JAX can differentiate, compile and batch it normally.
    """

    def __init__(self, param_db):
        self._param_db = param_db
        self._weights = {}
        self._terms = []
        self._term_for_st = {}

    @property
    def param_db(self):
        return self._param_db

    @property
    def terms(self):
        return list(self._terms)

    @property
    def weights(self):
        return dict(self._weights)

    def set_weight(self, score_type, weight):
        if not isinstance(score_type, ScoreType):
            raise TypeError("score_type must be a ScoreType")
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("score weights must be finite")
        if score_type not in _TERM_FOR_TYPE:
            raise NotImplementedError(f"JAX scoring does not support {score_type.name}")
        if score_type not in self._term_for_st and weight != 0:
            module, name = _TERM_FOR_TYPE[score_type]
            term_cls = getattr(
                importlib.import_module(
                    f"tmol_jax.score.{module}.{module}_energy_term"
                ),
                name,
            )
            term = term_cls(self._param_db)
            self._terms.append(term)
            for provided in term.score_types_provided():
                self._term_for_st[provided] = term
        self._weights[score_type] = weight

    def get_weight(self, score_type):
        return self._weights.get(score_type, 0.0)

    def active_score_types(self):
        return sorted(st for st, weight in self._weights.items() if weight != 0)

    def _setup(self, pose_stack):
        if pose_stack.packed_block_types is not None:
            from tmol_jax.chemical.constants import canonical_aa_3letter

            unsupported = {
                rt.name
                for rt in pose_stack.packed_block_types.active_block_types
                if rt.name3 not in canonical_aa_3letter
            }
            if unsupported:
                raise NotImplementedError(
                    "JAX scoring does not yet support: "
                    + ", ".join(sorted(unsupported))
                )
        if not np.isfinite(np.asarray(pose_stack.coords)).all():
            raise ValueError("pose coordinates must be finite before scoring setup")
        return tuple(
            term.setup_for_pose_stack(pose_stack, pose_stack.packed_block_types)
            for term in self._terms
        )

    def render_whole_pose_scoring_module(
        self, pose_stack, *, sum_terms=True, apply_weights=True
    ):
        """Return an unjitted coords -> scores callable with frozen configuration.

        Scores have shape [poses], or [active score types, poses] when
        sum_terms=False. Active types are ordered by ScoreType. Disabling
        apply_weights exposes those same channels without their weights.
        """
        channels = tuple(self.active_score_types())
        weights = dict(self._weights)
        states = self._setup(pose_stack)
        terms = tuple(copy.copy(term) for term in self._terms)
        entries = tuple(
            (term, term_state)
            for term, term_state in zip(terms, states)
            if any(weights.get(st, 0) for st in term.score_types_provided())
        )
        shape = pose_stack.coords.shape

        def score(coords):
            if coords.shape != shape or not jnp.issubdtype(coords.dtype, jnp.floating):
                raise ValueError(f"expected floating coordinates with shape {shape}")
            energies = {}
            for term, term_state in entries:
                energies.update(term(pose_stack, term_state, coords_override=coords))
            if not channels:
                return jnp.zeros(
                    (shape[0],) if sum_terms else (0, shape[0]), coords.dtype
                )
            values = jnp.stack(
                [
                    energies[st] * (weights[st] if apply_weights else 1)
                    for st in channels
                ]
            )
            return values.sum(axis=0) if sum_terms else values

        return score

    def jitted(self, pose_stack):
        return jax.jit(self.render_whole_pose_scoring_module(pose_stack))

    def jitted_grad(self, pose_stack):
        score = self.render_whole_pose_scoring_module(pose_stack)
        return jax.jit(jax.grad(lambda coords: score(coords).sum()))

    def jitted_value_and_grad(self, pose_stack):
        score = self.render_whole_pose_scoring_module(pose_stack)

        def value(coords):
            energies = score(coords)
            return energies.sum(), energies

        value_and_grad = jax.value_and_grad(value, has_aux=True)

        def apply(coords):
            (_, energies), gradient = value_and_grad(coords)
            return energies, gradient

        return jax.jit(apply)
