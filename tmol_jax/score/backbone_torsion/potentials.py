"""Low-level JAX potentials for backbone-torsion scoring.

Pure JAX port of the C++ kernels in
``tmol/score/backbone_torsion/potentials/potentials.hh``.

Ramachandran matches the upstream evaluator exactly: a 2-D degree-3
periodic B-spline interpolation of the pre-filtered coefficient grid
``[n_aa, n_phi, n_psi]`` (10-degree spacing, 36x36). The
coefficient-prefiltering is done once per database load in
:class:`tmol_jax.score.backbone_torsion.rama.RamaDatabaseAccess` via
:func:`tmol_jax.numeric.bspline.bspline_coefficients`; the per-residue
evaluator here simply evaluates the bicubic B-spline at the
fractional ``(phi, psi)`` grid index.

Omega is a harmonic spring around a backbone-dependent ideal
``omega0(phi, psi)`` with spring constant ``k(phi, psi)``. Upstream
tmol uses degree-3 periodic B-spline interpolation for both tables
(plus the ``log(sigma/6)`` baseline), and we follow suit here: the
``OmegaBBDepDatabaseAccess`` helper pre-filters each per-AA plane with
:func:`tmol_jax.numeric.bspline.bspline_coefficients` at load time,
and :func:`omega_potential` evaluates the bicubic B-spline on those
coefficients exactly like :func:`rama_potential` does.

A bilinear helper (:func:`bilinear_interpolate`) is retained for
backward compatibility, and a bicubic B-spline helper
(:func:`bicubic_bspline_interp`) is the shared evaluation primitive
used by both the Rama and omega potentials. Both functions are scalar
value functions of ``(phi, psi, ...)`` tensors; autograd flows through
via :func:`jax.grad`.
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.numeric.bspline import bspline_interpolate
from tmol_jax.types import Array

# 360 degrees, expressed as 2*pi radians — used to wrap phi/psi into the
# table domain regardless of whether the source angles are in [-pi, pi]
# or [0, 2*pi].
TWO_PI = 2.0 * jnp.pi


def wrap_angle(angle: Array, start: float) -> Array:
    """Wrap ``angle`` (radians) into ``[start, start + 2*pi)``.

    The rama/omega tables span a full 360 degree domain; inputs from
    :func:`tmol_jax.score.common.geom.dihedral_angle` are in ``[-pi, pi]``
    so we shift into the table's start offset before fractional indexing.
    """
    shifted = angle - start
    wrapped = jnp.mod(shifted, TWO_PI)
    return wrapped + start


def bilinear_interpolate(
    table: Array,
    phi: Array,
    psi: Array,
    bbstart: Array,
    bbstep: Array,
) -> Array:
    """Evaluate a 2-D periodic bilinear interpolation of ``table``.

    Parameters
    ----------
    table
        ``float[n_phi, n_psi]`` — energy/parameter grid, periodic in both
        axes (the ``n_phi``-th sample wraps to the 0th).
    phi, psi
        Scalar JAX arrays (radians).
    bbstart
        ``float[2]`` — (phi, psi) of the first grid sample (radians).
    bbstep
        ``float[2]`` — (phi, psi) spacing between grid samples (radians).

    Returns
    -------
    jnp.ndarray
        Scalar interpolated value.
    """
    n_phi = table.shape[0]
    n_psi = table.shape[1]

    # Fractional grid indices with periodic wrap (modulo table size).
    phi_wrapped = wrap_angle(phi, bbstart[0])
    psi_wrapped = wrap_angle(psi, bbstart[1])

    fx = (phi_wrapped - bbstart[0]) / bbstep[0]
    fy = (psi_wrapped - bbstart[1]) / bbstep[1]

    ix0 = jnp.floor(fx).astype(jnp.int32) % n_phi
    iy0 = jnp.floor(fy).astype(jnp.int32) % n_psi
    ix1 = (ix0 + 1) % n_phi
    iy1 = (iy0 + 1) % n_psi

    dx = fx - jnp.floor(fx)
    dy = fy - jnp.floor(fy)

    v00 = table[ix0, iy0]
    v01 = table[ix0, iy1]
    v10 = table[ix1, iy0]
    v11 = table[ix1, iy1]

    v0 = v00 * (1.0 - dy) + v01 * dy
    v1 = v10 * (1.0 - dy) + v11 * dy
    return v0 * (1.0 - dx) + v1 * dx


def bicubic_bspline_interp(
    coeffs: Array,
    phi: Array,
    psi: Array,
    bbstart: Array,
    bbstep: Array,
) -> Array:
    """Evaluate a 2-D periodic degree-3 B-spline at ``(phi, psi)``.

    This is the evaluation half of the Thévenaz-Blu-Unser B-spline
    algorithm: the input ``coeffs`` is the coefficient grid produced
    by a separable pair of causal / anti-causal IIR filters on the
    original samples (via :func:`tmol_jax.numeric.bspline.bspline_coefficients`).
    We then look up a 4x4 window and weight it by the 4 cubic B-spline
    basis functions in each axis, exactly as upstream tmol's
    ``ndspline<2, 3>::interpolate`` does.

    Parameters
    ----------
    coeffs
        ``float[n_phi, n_psi]`` pre-filtered B-spline coefficients
        (periodic). Same shape as the original sample grid.
    phi, psi
        Scalar dihedral angles (radians). Any convention works; we
        only need ``(phi - bbstart) / bbstep`` to land inside the
        periodic grid.
    bbstart, bbstep
        ``float[2]`` axis start and step (radians).

    Returns
    -------
    jnp.ndarray
        Scalar interpolated value. Differentiable in ``(phi, psi)``.
    """
    # Fractional grid indices. The angle's relation to bbstart modulo
    # 2*pi does not matter: the B-spline evaluator below wraps the
    # gather indices modulo the grid shape, so any real-valued
    # ``phi_idx`` / ``psi_idx`` produces the same periodic answer.
    phi_idx = (phi - bbstart[0]) / bbstep[0]
    psi_idx = (psi - bbstart[1]) / bbstep[1]
    X = jnp.stack([phi_idx, psi_idx])
    return bspline_interpolate(coeffs, X, degree=3)


def rama_potential(
    phi: Array,
    psi: Array,
    aa_type: Array,
    rama_coeffs: Array,
    bbstart: Array,
    bbstep: Array,
) -> Array:
    """Per-residue Rama energy: bicubic B-spline interpolation of the AA grid.

    Parameters
    ----------
    phi, psi
        Scalar dihedral angles (radians).
    aa_type
        Int index into the first axis of ``rama_coeffs``
        (``[n_aa, n_phi, n_psi]``). If negative, the potential returns
        0 — used to mask residue types without a tabulated Rama table
        (e.g. non-canonical residues).
    rama_coeffs
        ``float[n_aa, n_phi, n_psi]`` B-spline coefficient grid
        (produced by prefiltering the original energy tables — see
        :class:`tmol_jax.score.backbone_torsion.rama.RamaDatabaseAccess`).
    bbstart, bbstep
        Per-table axis start and step. Both are ``float[2]``; shared
        across all AAs in the typical upstream data.

    Returns
    -------
    jnp.ndarray
        Scalar energy (kcal/mol in the canonical tmol tables).
    """
    # Safe gather: clamp the AA index so we never read OOB even when the
    # caller passes -1 for "no table". We multiply by a mask below.
    safe_idx = jnp.where(aa_type >= 0, aa_type, 0)
    grid = rama_coeffs[safe_idx]
    val = bicubic_bspline_interp(grid, phi, psi, bbstart, bbstep)
    mask = (aa_type >= 0).astype(val.dtype)
    return val * mask


def omega_potential(
    phi: Array,
    psi: Array,
    omega: Array,
    aa_type: Array,
    k_table: Array,
    omega0_table: Array,
    bbstart: Array,
    bbstep: Array,
) -> Array:
    """Per-residue bb-dependent omega harmonic energy.

    The potential is

    .. math::

        E = k(\\phi, \\psi) \\cdot (\\omega - \\omega_0(\\phi, \\psi))^2,

    where ``k`` and ``omega0`` are evaluated via a degree-3 periodic
    B-spline of pre-filtered coefficient grids indexed by amino-acid
    type — matching upstream tmol's ``ndspline<2, 3>::interpolate``
    for both the spring-constant and equilibrium-mu tables. Omega is
    wrapped into ``[omega0 - pi, omega0 + pi]`` to avoid the
    360-degree seam.

    Parameters
    ----------
    phi, psi, omega
        Scalar dihedral angles (radians).
    aa_type
        Int index into the first axis of the tables. Negative values
        zero the result.
    k_table
        ``float[n_aa, n_phi, n_psi]`` B-spline coefficients for the
        spring-constant grid (kcal/mol/rad^2). Produced by
        :func:`tmol_jax.numeric.bspline.bspline_coefficients` applied
        separately to each AA plane.
    omega0_table
        ``float[n_aa, n_phi, n_psi]`` B-spline coefficients for the
        ideal omega grid (radians). Same pre-filter treatment.
    bbstart, bbstep
        Per-table axis start and step; ``float[2]``.

    Returns
    -------
    jnp.ndarray
        Scalar harmonic energy, zero when ``aa_type < 0``.
    """
    safe_idx = jnp.where(aa_type >= 0, aa_type, 0)
    k_grid = k_table[safe_idx]
    mu_grid = omega0_table[safe_idx]

    k = bicubic_bspline_interp(k_grid, phi, psi, bbstart, bbstep)
    mu = bicubic_bspline_interp(mu_grid, phi, psi, bbstart, bbstep)

    # Shortest signed distance between omega and its ideal, wrapped to
    # [-pi, pi]. This matches upstream's "if offset < -180, += 360"
    # normalization performed in degrees.
    diff = omega - mu
    wrapped = jnp.mod(diff + jnp.pi, TWO_PI) - jnp.pi
    energy = k * wrapped * wrapped

    mask = (aa_type >= 0).astype(energy.dtype)
    return energy * mask


__all__ = [
    "TWO_PI",
    "bicubic_bspline_interp",
    "bilinear_interpolate",
    "omega_potential",
    "rama_potential",
    "wrap_angle",
]
