"""Generic bonded scoring is zero for the currently supported canonical proteins."""

import jax.numpy as jnp

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class GenBondedEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.gen_torsions]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        return ()

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        return {ScoreType.gen_torsions: jnp.zeros(coords.shape[0], coords.dtype)}
