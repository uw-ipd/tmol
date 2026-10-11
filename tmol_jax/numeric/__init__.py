"""Pure-JAX port of ``tmol.numeric``.

Three small, stateless, jit/vmap-friendly numerical toolkits:

* :mod:`tmol_jax.numeric.rotations` -- 3-D rotation utilities
  (axis-angle, ZYX-Euler, quaternion conversions) built on stable
  :mod:`jax.numpy` primitives. Every function is differentiable.

* :mod:`tmol_jax.numeric.bspline` -- Thevenaz/Blu/Unser B-spline
  interpolation on regular periodic grids (the upstream routine used
  by :mod:`tmol.score.backbone_torsion` / :mod:`tmol.score.dunbrack`),
  plus pure Hermite-Catmull-Rom bicubic/tricubic interpolation. All
  jit-safe and differentiable via :func:`jax.grad`.

* :mod:`tmol_jax.numeric.cubic_hermite` -- direct port of the scalar
  cubic Hermite polynomial helpers in
  ``tmol/numeric/interpolation/cubic_hermite_polynomial.py`` (used by
  smooth knotted potentials elsewhere in tmol).

* :mod:`tmol_jax.numeric.dihedrals` -- vectorised dihedral-angle helper
  matching :func:`tmol.numeric.dihedrals.coord_dihedrals`.
"""

from __future__ import annotations

from tmol_jax.numeric.rotations import (
    rotation_matrix_from_axis_angle,
    euler_zyx_to_matrix,
    matrix_to_euler_zyx,
    quaternion_from_matrix,
    matrix_from_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
)
from tmol_jax.numeric.bspline import (
    BSplineInterpolation,
    bspline_coefficients,
    bspline_interpolate,
    bicubic_interp,
    tricubic_interp,
    catmull_rom_1d,
)
from tmol_jax.numeric.cubic_hermite import (
    interpolate_t,
    interpolate_dt,
    interpolate,
    interpolate_dx,
    interpolate_to_zero_t,
    interpolate_to_zero_dt,
    interpolate_to_zero,
    interpolate_to_zero_dx,
)
from tmol_jax.numeric.dihedrals import coord_dihedrals

__all__ = [
    # rotations
    "rotation_matrix_from_axis_angle",
    "euler_zyx_to_matrix",
    "matrix_to_euler_zyx",
    "quaternion_from_matrix",
    "matrix_from_quaternion",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
    # b-spline / bicubic
    "BSplineInterpolation",
    "bspline_coefficients",
    "bspline_interpolate",
    "bicubic_interp",
    "tricubic_interp",
    "catmull_rom_1d",
    # hermite polynomial helpers
    "interpolate_t",
    "interpolate_dt",
    "interpolate",
    "interpolate_dx",
    "interpolate_to_zero_t",
    "interpolate_to_zero_dt",
    "interpolate_to_zero",
    "interpolate_to_zero_dx",
    # dihedrals
    "coord_dihedrals",
]
