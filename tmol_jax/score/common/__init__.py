"""Shared math primitives for tmol_jax scoring.

Pure JAX ports of the C++ utilities in ``tmol/score/common``.
Derivatives are obtained via ``jax.grad`` — only value variants are
exposed here.
"""

from tmol_jax.score.common.count_pair import (
    MAX_COUNTPAIR_SEPARATION,
    connectivity_weight,
    pair_bonded_path_length_direct,
    pairwise_bonded_path_length,
    pose_atom_block_index,
    pose_atom_real_mask,
    scatter_per_atom,
)
from tmol_jax.score.common.cubic_hermite_polynomial import (
    interpolate,
    interpolate_t,
    interpolate_to_zero,
    interpolate_to_zero_t,
)
from tmol_jax.score.common.geom import (
    cos_interior_angle,
    dihedral_angle,
    distance,
    normalize,
    pt_interior_angle,
)

from tmol_jax.score.common.polynomial import bound_poly, poly_v

__all__ = [
    "MAX_COUNTPAIR_SEPARATION",
    "bound_poly",
    "connectivity_weight",
    "cos_interior_angle",
    "dihedral_angle",
    "distance",
    "interpolate",
    "interpolate_t",
    "interpolate_to_zero",
    "interpolate_to_zero_t",
    "normalize",
    "pair_bonded_path_length_direct",
    "pairwise_bonded_path_length",
    "pose_atom_block_index",
    "pose_atom_real_mask",
    "scatter_per_atom",
    "poly_v",
    "pt_interior_angle",
]
