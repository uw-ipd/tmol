"""Lennard-Jones and isotropic solvation over the current pose coordinates."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.count_pair import pairwise_bonded_path_length
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.ljlk.params import LJLKGlobal, LJTypeParams
from tmol_jax.score.ljlk.potentials import lj_atr_rep, lk_isotropic_pair
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass


@pytree_dataclass
class _LJLKState:
    parameters: LJTypeParams
    global_parameters: LJLKGlobal
    bonded_path_length: jax.Array
    atom_is_real: jax.Array


class LJLKEnergyTerm(EnergyTerm):
    """Fixed topology, differentiable coordinates; no stale spatial neighbors."""

    def score_types_provided(self):
        return [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        indices = np.full(pose_stack.coords.shape[:2], -1, dtype=np.int32)
        block_types = np.asarray(pose_stack.block_type_ind)
        offsets = np.asarray(pose_stack.block_coord_offset)
        atom_types = np.asarray(pbt.atom_types)
        for pose, block in zip(*np.nonzero(block_types >= 0)):
            bt = block_types[pose, block]
            offset = offsets[pose, block]
            n_atoms = len(pbt.active_block_types[bt].atoms)
            indices[pose, offset : offset + n_atoms] = atom_types[bt, :n_atoms]
        parameters = LJTypeParams.from_database(
            self.param_db.scoring.ljlk, self.param_db.chemical
        )
        return _LJLKState(
            parameters.gather(jnp.asarray(indices)),
            LJLKGlobal.from_database(self.param_db.scoring.ljlk),
            pairwise_bonded_path_length(pose_stack, pbt),
            jnp.asarray(indices >= 0),
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        if coords.dtype == jnp.float32 and jax.default_backend() == "gpu":
            from tmol_jax.score.ljlk import cuda

            if cuda.available():
                values = cuda.score(term_state, coords)
                return dict(zip(self.score_types_provided(), values.T))
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        distance_squared = jnp.sum(diff * diff, axis=-1)
        nonzero = distance_squared > 0
        # Coincident points have undefined radial direction. Keep inactive
        # self-pairs finite, while evaluating the potential's zero-distance limit.
        distance = jnp.where(
            nonzero, jnp.sqrt(jnp.where(nonzero, distance_squared, 1)), 0
        )
        state = term_state
        first = jax.tree.map(lambda x: x[:, :, None], state.parameters)
        second = jax.tree.map(lambda x: x[:, None, :], state.parameters)
        attractive, repulsive = lj_atr_rep(
            distance, state.bonded_path_length, first, second, state.global_parameters
        )
        solvation = lk_isotropic_pair(
            distance, state.bonded_path_length, first, second, state.global_parameters
        )
        mask = state.atom_is_real[:, :, None] & state.atom_is_real[:, None, :]
        mask &= jnp.triu(jnp.ones(distance.shape[-2:], dtype=bool), 1)
        return {
            score_type: jnp.where(mask, value, 0).sum(axis=(-1, -2))
            for score_type, value in zip(
                self.score_types_provided(), (attractive, repulsive, solvation)
            )
        }
