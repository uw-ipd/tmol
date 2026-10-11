"""B-spline and cubic-Hermite interpolation in pure JAX.

This module provides two complementary tools:

1. **Thevenaz/Blu/Unser B-splines** (:func:`bspline_coefficients`,
   :func:`bspline_interpolate`, :class:`BSplineInterpolation`) -- a
   line-for-line port of the C++ header
   ``tmol/numeric/bspline_compiled/bspline.hh`` used by upstream
   tmol's Ramachandran and Dunbrack potentials. The same math; the
   only change is that the causal / anti-causal IIR filters that
   produce spline coefficients are expressed as :func:`jax.lax.scan`
   recurrences and vectorised with :func:`jax.vmap` over the remaining
   axes. Periodic boundary conditions; supports degree-2..5 in
   2/3/4 spatial dimensions.

2. **Hermite/Catmull-Rom bicubic / tricubic interpolation**
   (:func:`bicubic_interp`, :func:`tricubic_interp`) -- a lightweight
   alternative for consumers that do not need to pre-fit spline
   coefficients. Operates directly on the sample grid with periodic
   wrap-around. Every operation is differentiable, so
   ``jax.grad(bicubic_interp, argnums=(1, 2))`` yields the exact
   analytic gradient in ``(u, v)``.

The upstream ``BSplineInterpolation.from_coordinates`` /
``.interpolate`` two-step API is preserved unchanged so callers in
``tmol_jax.score.backbone_torsion`` / ``tmol_jax.score.dunbrack`` can
mirror their torch counterparts.

Reference
---------
Thevenaz, Blu, and Unser. *Interpolation revisited [medical images
application]*. IEEE TMI 19(7), 2000.
http://bigwww.epfl.ch/publications/thevenaz0002.pdf
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:  # Prefer the package-level pytree decorator when importable.
    from tmol_jax.types import pytree_dataclass  # type: ignore
except (
    Exception
):  # pragma: no cover - fallback for old Python where tmol_jax.types fails to import
    import attrs

    def pytree_dataclass(cls):
        """Minimal stand-in: an ``attrs``-defined class registered as a
        pytree with array children and scalar aux data.

        Keeps this module importable even if the top-level ``tmol_jax.types``
        module is broken on the current interpreter. Semantics match the
        version in ``tmol_jax.types``.
        """

        if not hasattr(cls, "__attrs_attrs__"):
            cls = attrs.define(slots=False)(cls)
        field_names = tuple(f.name for f in cls.__attrs_attrs__)

        def tree_flatten(obj):
            values = [getattr(obj, n) for n in field_names]
            child_mask = tuple(isinstance(v, (jax.Array, np.ndarray)) for v in values)
            children = tuple(v for v, m in zip(values, child_mask) if m)
            aux_values = tuple(v for v, m in zip(values, child_mask) if not m)
            return children, (child_mask, aux_values)

        def tree_unflatten(aux, children):
            child_mask, aux_values = aux
            c_iter = iter(children)
            a_iter = iter(aux_values)
            kw = {}
            for name, is_child in zip(field_names, child_mask):
                kw[name] = next(c_iter) if is_child else next(a_iter)
            return cls(**kw)

        jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
        return cls


# ---------------------------------------------------------------------------
# Helpers: poles for B-splines of various degrees
# ---------------------------------------------------------------------------


def _get_poles(degree: int) -> Tuple[float, ...]:
    """Return the tuple of real poles for a B-spline of the given degree.

    Degrees 2..5 supported; matches the upstream C++ ``_get_poles``.
    """
    if degree == 2:
        return (np.sqrt(8.0) - 3.0,)
    if degree == 3:
        return (np.sqrt(3.0) - 2.0,)
    if degree == 4:
        return (
            np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
            np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0,
        )
    if degree == 5:
        return (
            np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0))
            + np.sqrt(105.0 / 4.0)
            - 13.0 / 2.0,
            np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0))
            - np.sqrt(105.0 / 4.0)
            - 13.0 / 2.0,
        )
    raise ValueError(f"B-spline degree must be in 2..5 (got {degree}).")


# ---------------------------------------------------------------------------
# 1D causal / anti-causal coefficient conversion (Thevenaz eqns 2-3)
# ---------------------------------------------------------------------------


def _init_causal_coeff(line: jax.Array, pole: float) -> jax.Array:
    """Periodic boundary init for the causal recursion."""
    N = line.shape[0]
    tol = 4.0 * np.finfo(np.float64).eps
    horiz = int(np.ceil(np.log(tol) / np.log(np.abs(pole))))

    if horiz < N:
        # line[0] += sum_{i=1}^{horiz-1} pole^i * line[N-i]
        i = jnp.arange(1, horiz)  # static length -> static shape
        zn = pole**i
        contributions = zn * line[N - i]
        return line.at[0].add(jnp.sum(contributions))
    else:
        # Closed-form for short signals, line[0] / (1 - pole^N)
        i = jnp.arange(1, N)
        zn = pole**i
        zn_N = pole**N
        contributions = zn * line[N - i]
        new0 = (line[0] + jnp.sum(contributions)) / (1.0 - zn_N)
        return line.at[0].set(new0)


def _init_anticausal_coeff(line: jax.Array, pole: float) -> jax.Array:
    """Periodic boundary init for the anti-causal recursion."""
    N = line.shape[0]
    tol = 4.0 * np.finfo(np.float64).eps
    horiz = int(np.ceil(np.log(tol) / np.log(np.abs(pole))))

    if horiz < N:
        i = jnp.arange(0, horiz)
        zn = pole ** (i + 1)
        contributions = zn * line[i]
        new_last = (line[N - 1] + jnp.sum(contributions)) * (-pole)
        return line.at[N - 1].set(new_last)
    else:
        i = jnp.arange(0, N)
        zn = pole ** (i + 1)
        zn_N = pole**N
        contributions = zn * line[i]
        new_last = (line[N - 1] + jnp.sum(contributions)) * (-pole) / (1.0 - zn_N)
        return line.at[N - 1].set(new_last)


def _convert_interp_coeffs_1d(line: jax.Array, poles: Tuple[float, ...]) -> jax.Array:
    """In-place (functional) conversion of a 1-D stripe of samples to
    B-spline coefficients. Matches ``_convert_interp_coeffs`` in
    ``bspline.hh``.
    """

    N = line.shape[0]
    if N == 1:
        return line

    # Overall gain lambda = prod_i (1 - p_i) * (1 - 1/p_i)
    lam = 1.0
    for p in poles:
        lam *= (1.0 - p) * (1.0 - 1.0 / p)
    line = line * lam

    def apply_pole(line, pole):
        # Causal filter: c[j] = s[j] + pole * c[j-1]
        line = _init_causal_coeff(line, pole)

        def causal_step(prev, cur):
            out = cur + pole * prev
            return out, out

        _, tail = jax.lax.scan(causal_step, line[0], line[1:])
        line = jnp.concatenate([line[:1], tail])

        # Anti-causal filter: c[j] = pole * (c[j+1] - c[j])
        line = _init_anticausal_coeff(line, pole)

        def anticausal_step(prev, cur):
            out = pole * (prev - cur)
            return out, out

        rev = line[:-1][::-1]
        _, tail_rev = jax.lax.scan(anticausal_step, line[-1], rev)
        line = jnp.concatenate([tail_rev[::-1], line[-1:]])
        return line

    for p in poles:
        line = apply_pole(line, p)
    return line


def _apply_along_axis(
    coeffs: jax.Array, axis: int, poles: Tuple[float, ...]
) -> jax.Array:
    """Run :func:`_convert_interp_coeffs_1d` along ``axis`` of ``coeffs``.

    Achieved by moving the target axis to the front, vmapping over all
    remaining axes, and restoring the original layout.
    """

    coeffs = jnp.moveaxis(coeffs, axis, 0)
    shape = coeffs.shape
    flat = coeffs.reshape(shape[0], -1)

    # One call per stripe -- vmap over the flattened trailing dim.
    stripe_fn = partial(_convert_interp_coeffs_1d, poles=poles)
    flat_out = jax.vmap(stripe_fn, in_axes=1, out_axes=1)(flat)

    out = flat_out.reshape(shape)
    return jnp.moveaxis(out, 0, axis)


# ---------------------------------------------------------------------------
# Public: coefficient precomputation
# ---------------------------------------------------------------------------


def bspline_coefficients(samples: jax.Array, degree: int = 3) -> jax.Array:
    """Convert an N-dim tensor of samples to B-spline coefficients.

    The output has the same shape as ``samples`` (B-splines do not
    expand the grid). Periodic boundary conditions are applied along
    every axis.

    Parameters
    ----------
    samples : jax.Array
        Tensor of shape ``[N_0, N_1, ..., N_{D-1}]`` with ``D in 2..4``
        spatial dimensions. ``float32`` or ``float64``.
    degree : int, default 3
        B-spline order (2, 3, 4, or 5). Degree 3 matches the upstream
        default and is the canonical choice for Rama/Dunbrack.

    Returns
    -------
    jax.Array
        Coefficient tensor, same shape as ``samples``.
    """

    coeffs = jnp.asarray(samples)
    if coeffs.ndim < 2 or coeffs.ndim > 4:
        raise ValueError(
            "B-spline coefficient computation supports 2-4 dims "
            f"(got ndim={coeffs.ndim})."
        )
    poles = _get_poles(degree)
    for axis in range(coeffs.ndim):
        coeffs = _apply_along_axis(coeffs, axis, poles)
    return coeffs


# ---------------------------------------------------------------------------
# Weights and derivatives for interpolation
# ---------------------------------------------------------------------------


def _weights_1d(frac: jax.Array, degree: int) -> jax.Array:
    """Per-dimension spline weights for the (DEGREE+1) support cells.

    ``frac`` is the fractional offset (shape ``[...]``); the returned
    array has shape ``[..., DEGREE+1]``. Directly mirrors the
    ``_get_weights`` function in ``bspline.hh``.
    """

    f = jnp.asarray(frac)
    if degree == 2:
        w1 = 0.75 - f * f
        w2 = 0.5 * (f - w1 + 1.0)
        w0 = 1.0 - w1 - w2
        return jnp.stack([w0, w1, w2], axis=-1)
    if degree == 3:
        w3 = (1.0 / 6.0) * f * f * f
        w0 = (1.0 / 6.0) + 0.5 * f * (f - 1.0) - w3
        w2 = f + w0 - 2.0 * w3
        w1 = 1.0 - w0 - w2 - w3
        return jnp.stack([w0, w1, w2, w3], axis=-1)
    if degree == 4:
        t = (1.0 / 6.0) * f * f
        t0 = f * (t - 11.0 / 24.0)
        t1 = 19.0 / 96.0 + f * f * (0.25 - t)
        w0 = (1.0 / 24.0) * (0.5 - f) ** 3
        w1 = t1 + t0
        w3 = t1 - t0
        w4 = w0 + t0 + 0.5 * f
        w2 = 1.0 - w0 - w1 - w3 - w4
        return jnp.stack([w0, w1, w2, w3, w4], axis=-1)
    if degree == 5:
        f2 = f * f
        w5 = (1.0 / 120.0) * f2 * f * f2
        w4d = (f2 - f) ** 2
        w_ = f - 0.5
        t = f2 * (f2 - 3.0)
        w0 = (1.0 / 24.0) * (0.2 + f2 + w4d) - w5
        t0 = (1.0 / 24.0) * (f2 * (f2 - 5.0) + 46.0 / 5.0)
        t1 = (-1.0 / 12.0) * w_ * (t + 4.0)
        w2 = t0 + t1
        w3 = t0 - t1
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t)
        t1 = (1.0 / 24.0) * w_ * (w4d - f2 - 5.0)
        w1 = t0 + t1
        w4 = t0 - t1
        return jnp.stack([w0, w1, w2, w3, w4, w5], axis=-1)
    raise ValueError(f"Unsupported B-spline degree {degree}.")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate_single(coeffs: jax.Array, X: jax.Array, degree: int) -> jax.Array:
    """Interpolate ``coeffs`` at a single N-D coordinate ``X``.

    Returns a scalar. Gradient w.r.t. ``X`` is available through
    :func:`jax.grad`; we don't return it explicitly because a manual
    hand-derived gradient is unnecessary in JAX.
    """

    ndim = coeffs.ndim
    # Integer base index in each dim and fractional offset.
    idx0 = jnp.floor(X).astype(jnp.int32)  # [ndim]
    frac = X - idx0.astype(X.dtype)
    idx_base = idx0 - (degree // 2)  # support starts here

    # Per-dim weights: [ndim, DEGREE+1]
    wts = jax.vmap(lambda fi: _weights_1d(fi, degree))(frac)

    # Outer product of weights -> full [DEGREE+1]^ndim tensor.
    w = wts[0]
    for d in range(1, ndim):
        w = w[..., None] * wts[d]

    # Gather coefficient neighbourhood via periodic wrap-around.
    k = degree + 1
    offsets = np.arange(k)  # static python-range -> static shape

    # Build a broadcast of [..., ndim] indices covering the ndim-D window.
    per_dim_indices = []
    for d in range(ndim):
        # shape: broadcastable to ndim axes of length k each
        shape = [1] * ndim
        shape[d] = k
        per_dim_indices.append(
            jnp.reshape(
                jnp.mod(idx_base[d] + offsets, coeffs.shape[d]),
                shape,
            )
        )

    neighbourhood = coeffs[tuple(per_dim_indices)]
    # Weighted sum.
    return jnp.sum(w * neighbourhood)


def bspline_interpolate(coeffs: jax.Array, X: jax.Array, degree: int = 3) -> jax.Array:
    """Interpolate (pre-computed) B-spline coefficients at points.

    Parameters
    ----------
    coeffs : jax.Array
        Coefficient tensor produced by :func:`bspline_coefficients`.
        Shape ``[N_0, ..., N_{D-1}]`` with ``D = coeffs.ndim``.
    X : jax.Array
        Query coordinates. Either a single point of shape ``[D]`` or a
        batch of shape ``[..., D]``. Each coordinate is in grid-cell
        units (periodic): ``X[..., d] in R`` with integer values
        aligning on grid nodes.
    degree : int, default 3

    Returns
    -------
    jax.Array
        Interpolated values with shape ``X.shape[:-1]``.
    """
    X = jnp.asarray(X)
    if X.ndim == 1:
        return _interpolate_single(coeffs, X, degree)

    # Batched: vmap over all leading axes.
    fn = partial(_interpolate_single, coeffs, degree=degree)
    # Collapse the leading dims to a single batch, then restore.
    lead = X.shape[:-1]
    flat = X.reshape(-1, X.shape[-1])
    out = jax.vmap(fn)(flat)
    return out.reshape(lead)


# ---------------------------------------------------------------------------
# Stateful class: preserves the upstream two-phase API
# ---------------------------------------------------------------------------


@pytree_dataclass
class BSplineInterpolation:
    """Container that bundles precomputed spline coefficients.

    Mirrors :class:`tmol.numeric.bspline.BSplineInterpolation`: construct
    via :meth:`from_coordinates` once, then call :meth:`interpolate`
    many times.

    The object is a registered JAX pytree so it can be passed through
    :func:`jax.jit` / :func:`jax.vmap` boundaries, and it is
    ``@pytree_dataclass``-frozen.
    """

    coeffs: jax.Array
    degree: int = 3

    @classmethod
    def from_coordinates(cls, coords, degree: int = 3) -> "BSplineInterpolation":
        """Build coefficients from an N-D sample grid."""
        coeffs = bspline_coefficients(jnp.asarray(coords), degree=degree)
        return cls(coeffs=coeffs, degree=degree)

    def interpolate(self, X: jax.Array) -> jax.Array:
        """Interpolate at the given coordinates."""
        return bspline_interpolate(self.coeffs, X, degree=self.degree)


# ---------------------------------------------------------------------------
# Hermite / Catmull-Rom bicubic & tricubic on raw sample grids
# ---------------------------------------------------------------------------


def catmull_rom_1d(
    p_m1: jax.Array, p_0: jax.Array, p_1: jax.Array, p_2: jax.Array, t: jax.Array
) -> jax.Array:
    """1-D Catmull-Rom cubic interpolation at fractional position ``t in [0, 1]``.

    ``p_m1``, ``p_0``, ``p_1``, ``p_2`` are the four samples surrounding
    ``t`` (``t == 0`` -> ``p_0``, ``t == 1`` -> ``p_1``). Uses the
    standard Catmull-Rom tangents
    ``m_0 = (p_1 - p_m1) / 2`` and ``m_1 = (p_2 - p_0) / 2``; the cubic
    Hermite polynomial with these tangents interpolates through ``p_0``
    and ``p_1`` with C^1 continuity.
    """
    # Pre-expanded Catmull-Rom polynomial.
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p_0)
        + (-p_m1 + p_1) * t
        + (2.0 * p_m1 - 5.0 * p_0 + 4.0 * p_1 - p_2) * t2
        + (-p_m1 + 3.0 * p_0 - 3.0 * p_1 + p_2) * t3
    )


def _wrap(idx: jax.Array, N: int) -> jax.Array:
    """Periodic wrap-around for an integer index (concrete ``N``)."""
    return jnp.mod(idx, N)


def bicubic_interp(grid: jax.Array, u: jax.Array, v: jax.Array) -> jax.Array:
    """Catmull-Rom bicubic interpolation on a 2-D periodic ``grid``.

    Parameters
    ----------
    grid : jax.Array
        ``[H, W]`` array of samples. Indexing is ``grid[h, w]`` with
        ``h`` corresponding to coordinate ``u`` and ``w`` to ``v``.
    u, v : jax.Array (scalar)
        Fractional grid coordinates. ``u == 0`` samples ``grid[0, :]``;
        ``u == H`` wraps back to ``grid[0, :]``. Same for ``v`` vs
        ``W``.

    Returns
    -------
    jax.Array
        Scalar interpolated value. Differentiable in ``u`` / ``v``.
    """

    H, W = grid.shape
    u = jnp.asarray(u)
    v = jnp.asarray(v)

    iu = jnp.floor(u).astype(jnp.int32)
    iv = jnp.floor(v).astype(jnp.int32)
    fu = u - iu.astype(u.dtype)
    fv = v - iv.astype(v.dtype)

    # Four rows (ih = iu-1, iu, iu+1, iu+2) and, for each, four columns.
    def row_at(du: int) -> jax.Array:
        ih = _wrap(iu + du, H)
        iw_m1 = _wrap(iv - 1, W)
        iw_0 = _wrap(iv, W)
        iw_1 = _wrap(iv + 1, W)
        iw_2 = _wrap(iv + 2, W)
        return catmull_rom_1d(
            grid[ih, iw_m1], grid[ih, iw_0], grid[ih, iw_1], grid[ih, iw_2], fv
        )

    r_m1 = row_at(-1)
    r_0 = row_at(0)
    r_1 = row_at(1)
    r_2 = row_at(2)
    return catmull_rom_1d(r_m1, r_0, r_1, r_2, fu)


def tricubic_interp(
    grid: jax.Array, u: jax.Array, v: jax.Array, w: jax.Array
) -> jax.Array:
    """Catmull-Rom tricubic interpolation on a 3-D periodic ``grid``.

    Parameters
    ----------
    grid : ``[D, H, W]`` array
    u, v, w : scalar fractional coordinates (periodic)

    Returns
    -------
    Scalar interpolated value, differentiable in ``(u, v, w)``.
    """

    D, H, W = grid.shape
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    w = jnp.asarray(w)

    iu = jnp.floor(u).astype(jnp.int32)
    iv = jnp.floor(v).astype(jnp.int32)
    iw = jnp.floor(w).astype(jnp.int32)
    fu = u - iu.astype(u.dtype)
    fv = v - iv.astype(v.dtype)
    fw = w - iw.astype(w.dtype)

    def slab_at(du: int) -> jax.Array:
        # Treat the slab at offset ``du`` along the first axis as a 2-D
        # bicubic over (iv, iw) and reuse bicubic_interp locally.
        id_ = _wrap(iu + du, D)
        slab = grid[id_]  # [H, W]
        # We inline the 2-D routine to avoid re-wrapping iv twice.
        iv_m1 = _wrap(iv - 1, H)
        iv_0 = _wrap(iv, H)
        iv_1 = _wrap(iv + 1, H)
        iv_2 = _wrap(iv + 2, H)
        iw_m1 = _wrap(iw - 1, W)
        iw_0 = _wrap(iw, W)
        iw_1 = _wrap(iw + 1, W)
        iw_2 = _wrap(iw + 2, W)

        def row_at(ih):
            return catmull_rom_1d(
                slab[ih, iw_m1], slab[ih, iw_0], slab[ih, iw_1], slab[ih, iw_2], fw
            )

        r_m1 = row_at(iv_m1)
        r_0 = row_at(iv_0)
        r_1 = row_at(iv_1)
        r_2 = row_at(iv_2)
        return catmull_rom_1d(r_m1, r_0, r_1, r_2, fv)

    s_m1 = slab_at(-1)
    s_0 = slab_at(0)
    s_1 = slab_at(1)
    s_2 = slab_at(2)
    return catmull_rom_1d(s_m1, s_0, s_1, s_2, fu)


__all__ = [
    "BSplineInterpolation",
    "bspline_coefficients",
    "bspline_interpolate",
    "bicubic_interp",
    "tricubic_interp",
    "catmull_rom_1d",
]
