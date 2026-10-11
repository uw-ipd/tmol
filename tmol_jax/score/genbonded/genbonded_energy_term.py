"""Canonical proteins have no generic atom types and zero generic energy."""

import jax.numpy as jnp

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class GenBondedEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.gen_torsions]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        generic_types = self.param_db.scoring.genbonded.atom_hierarchy
        if any(
            atom.atom_type in generic_types
            for residue in pbt.active_block_types
            for atom in residue.atoms
        ):
            raise NotImplementedError("JAX generic ligand torsions are not implemented")
        return ()

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        return {ScoreType.gen_torsions: jnp.zeros(coords.shape[0], coords.dtype)}
