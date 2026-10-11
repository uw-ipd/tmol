"""Vector geometry helpers in pure JAX.

All functions are vectorized: inputs are ``[..., 3]`` arrays and the
leading batch dimensions are preserved. Derivatives are available via
``jax.grad``; only value functions are ported from ``geom.hh``.

Autodiff safety
---------------
Upstream ``tmol`` hand-writes every adjoint kernel with explicit
case-splits for degenerate geometry (colinear atoms, zero-length
bonds, perfect 180° angles). Our port uses ``jax.grad`` through
``jnp.where``, which autodiffs through BOTH branches of every
``jnp.where`` — so any division by zero, ``sqrt(0)``, ``arccos(±1)``,
``atan2(0, 0)``, or ``norm(0)`` on the "masked-out" branch still
backpropagates NaN/inf cotangents.

Every primitive below uses the double-``where`` pattern: the
denominator (or the radicand of a sqrt, or the argument of an
``atan2``) is swapped for a safe placeholder on the degenerate
branch BEFORE the unsafe op runs, then the output is rewritten to
the correct primal value AFTER. The primal path sees the true value;
the cotangent path sees a finite partial derivative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tmol_jax.types import Array

# ---------------------------------------------------------------------------#
# Safe primitives
# ---------------------------------------------------------------------------#
#
# The ``eps`` values below are chosen so that ``sq > eps`` separates
# "exact zero" (padded / degenerate slots) from any physically
# plausible geometry. For single-precision (float32) coordinates,
# ``1e-12`` on a squared length corresponds to a vector magnitude
# of ``1e-6`` Å, which is below the resolution of any PDB input and
# below the noise floor of any optimizer step. For atan2 we use
# ``1e-30`` on ``x^2 + y^2`` because ``atan2`` only becomes ill-
# conditioned near the exact origin.

_EPS_SQ = 1e-12
_EPS_ATAN2 = 1e-30
_EPS_ACOS = 1e-7


def _safe_norm(
    v: Array, axis: int = -1, keepdims: bool = False, eps: float = _EPS_SQ
) -> Array:
    """NaN-safe Euclidean norm along ``axis``.

    Returns ``0`` on the degenerate branch (``|v|^2 <= eps``) and the
    true norm otherwise. Gradients are finite everywhere — the
    double-``where`` swaps the zero radicand for ``1.0`` inside the
    ``sqrt`` so the ``1/(2*sqrt(x))`` cotangent stays bounded.
    """
    sq = jnp.sum(v * v, axis=axis, keepdims=keepdims)
    safe = sq > eps
    return jnp.where(safe, jnp.sqrt(jnp.where(safe, sq, 1.0)), 0.0)


def _safe_normalize(v: Array, axis: int = -1, eps: float = _EPS_SQ) -> Array:
    """NaN-safe unit vector along ``axis``.

    Returns the zero vector on the degenerate branch; otherwise
    ``v / |v|``. Both primal and cotangent are finite — the
    double-``where`` on the norm makes the division safe, and the
    outer ``where`` restores the correct primal.
    """
    sq = jnp.sum(v * v, axis=axis, keepdims=True)
    safe = sq > eps
    norm = jnp.where(safe, jnp.sqrt(jnp.where(safe, sq, 1.0)), 1.0)
    return jnp.where(safe, v / norm, jnp.zeros_like(v))


def _safe_acos(x: Array, eps: float = _EPS_ACOS) -> Array:
    """NaN-safe ``arccos``.

    ``arccos`` has derivative ``-1/sqrt(1 - x^2)``, unbounded at
    ``|x| == 1``. Clipping strictly inside ``[-1 + eps, 1 - eps]``
    keeps the gradient finite without perturbing any physically
    meaningful value (cosines reach the endpoints only at exact
    colinearity).
    """
    return jnp.arccos(jnp.clip(x, -1.0 + eps, 1.0 - eps))


def _safe_atan2(y: Array, x: Array, eps: float = _EPS_ATAN2) -> Array:
    """NaN-safe ``arctan2``.

    ``arctan2`` has derivative ``(-y, x) / (x^2 + y^2)``, NaN at the
    origin. Swap both args to ``(x=1, y=0)`` on the degenerate branch
    so the partial derivative is ``(0, 1) / 1 = (0, 1)`` (finite),
    then rewrite the primal to ``0`` — the correct limiting value
    for our call sites, all of which treat ``atan2(0, 0)`` as an
    undefined angle that is always multiplied by a zero weight
    downstream.
    """
    degenerate = (y * y + x * x) < eps
    y_safe = jnp.where(degenerate, 0.0, y)
    x_safe = jnp.where(degenerate, 1.0, x)
    return jnp.where(degenerate, jnp.zeros_like(y_safe), jnp.arctan2(y_safe, x_safe))


# ---------------------------------------------------------------------------#
# Public API
# ---------------------------------------------------------------------------#


def distance(a: Array, b: Array) -> Array:
    """Euclidean distance between points ``a`` and ``b`` along the last axis.

    NaN-safe under ``jax.grad``: ``d/dx sqrt(x) = 1/(2 sqrt(x))`` is
    ``inf`` at ``x = 0``, which poisons every cotangent that flows
    through it. The :func:`_safe_norm` primitive handles this with
    the double-``where`` trick.
    """
    return _safe_norm(a - b, axis=-1)


def normalize(v: Array) -> Array:
    """Return ``v`` scaled to unit length along its last axis.

    NaN-safe for zero-length inputs: see :func:`_safe_normalize`.
    """
    return _safe_normalize(v, axis=-1)


def cos_interior_angle(u: Array, v: Array) -> Array:
    """Cosine of the angle between vectors ``u`` and ``v``.

    Safe against zero-length inputs (returns 0 on the degenerate
    branch) and clipped strictly inside ``(-1, 1)`` so that any
    downstream ``arccos`` stays differentiable at colinearity
    (its derivative ``-1/sqrt(1 - x^2)`` is unbounded at the
    endpoints).
    """
    dot = jnp.sum(u * v, axis=-1)
    u_sq = jnp.sum(u * u, axis=-1)
    v_sq = jnp.sum(v * v, axis=-1)
    safe_u = u_sq > _EPS_SQ
    safe_v = v_sq > _EPS_SQ
    # Double-where on both sqrts — the cotangent of ``sqrt(0)`` is
    # inf, and would propagate into the product ``u_norm * v_norm``
    # even after ``safe_u & safe_v`` gates the quotient.
    u_norm = jnp.where(safe_u, jnp.sqrt(jnp.where(safe_u, u_sq, 1.0)), 1.0)
    v_norm = jnp.where(safe_v, jnp.sqrt(jnp.where(safe_v, v_sq, 1.0)), 1.0)
    denom_safe = safe_u & safe_v
    # Double-where on the divide — if the denominator is zero on the
    # "unsafe" branch, ``dot / denom`` still contributes to the JVP.
    denom = jnp.where(denom_safe, u_norm * v_norm, 1.0)
    cos = jnp.where(denom_safe, dot / denom, 0.0)
    # Strictly interior clip — arccos gradient is ``-1/sqrt(1-c^2)``
    # and blows up at ``|c| == 1``; the epsilon keeps grads finite
    # without measurably affecting the value.
    return jnp.clip(cos, -1.0 + _EPS_ACOS, 1.0 - _EPS_ACOS)


def _interior_angle_atan2(u: Array, v: Array) -> Array:
    """Interior angle between vectors ``u`` and ``v`` via ``atan2``.

    Mirrors the upstream ``interior_angle<Real>::V`` in ``geom.hh``::

        CR = u x v
        z_unit = CR / |CR|
        angle = 2 * atan2(CR . z_unit, |u| * |v| + u . v)

    Numerically stable across ``[0, pi]`` (arccos loses precision near
    the endpoints where ``|u . v|`` approaches ``|u| * |v|``). The
    factor of 2 and the ``|u||v| + u.v`` denominator together use the
    identity ``tan(theta/2) = sin(theta) / (1 + cos(theta))`` to avoid
    cancellation near ``theta = pi``.

    NaN-safety via :func:`_safe_norm` and :func:`_safe_atan2`:

    * ``_safe_norm`` handles the zero-length cross product (colinear
      ``u, v``) and the zero-length ``u`` / ``v`` themselves.
    * ``_safe_atan2`` handles the fully-degenerate case ``u == 0`` or
      ``v == 0`` where both ``atan2`` arguments are zero. This shows
      up at score time because padded subgraph slots gather atom
      index 0 three times over, so ``u = v = 0`` for those entries;
      the forward ``jnp.where(mask, ...)`` zeros their energy
      contribution but the cotangent still flows through both
      branches and would pollute the gradient at atom 0.
    """
    cr = jnp.cross(u, v)
    cr_mag = _safe_norm(cr, axis=-1)
    u_norm = _safe_norm(u, axis=-1)
    v_norm = _safe_norm(v, axis=-1)
    dot = jnp.sum(u * v, axis=-1)
    # Upstream: 2 * atan2(CR . z_unit, |u|*|v| + u.v) — here ``CR . z_unit``
    # simplifies to ``|CR|`` since z_unit = CR/|CR|.
    return 2.0 * _safe_atan2(cr_mag, u_norm * v_norm + dot)


@jax.custom_jvp
def pt_interior_angle(a: Array, b: Array, c: Array) -> Array:
    """Interior angle at point ``b`` formed by segments ``b-a`` and ``b-c``.

    Returns the angle in radians. Uses the upstream ``atan2`` formulation
    via :func:`_interior_angle_atan2` for numerical stability across
    the full ``[0, pi]`` range — especially important for hbond
    scoring, where AHD angles can sit close to ``pi`` for well-aligned
    donors and the ``arccos`` path would lose precision there.

    Gradients are computed via :func:`jax.custom_jvp` against the
    upstream analytical adjoint in ``interior_angle::V_dV`` in
    ``geom.hh``. Because the autograd path through ``2 * atan2(|CR|,
    |u|*|v| + u.v)`` accumulates roundoff that is bit-distinct from
    upstream's closed-form ``(u/|u|) x z_unit / |u|`` adjoint, we
    skip autograd entirely and emit the analytic ``dV/dA``, ``dV/dB``,
    ``dV/dC`` directly. This drops the cart_angles gradient error from
    ~1e-3 to float-precision roundoff on 1ubq.
    """
    # Upstream ``pt_interior_angle::V`` in ``geom.hh`` calls
    # ``interior_angle::V(a-b, c-b)`` which uses the 2*atan2 identity
    # rather than ``arccos(cos(theta))``. The atan2 form avoids the
    # precision blowup that ``arccos`` has near 0 and pi; for hbonds
    # AHD is typically close to pi and BAH is typically close to 2pi/3,
    # so matching upstream's numerics gives ~0.05 energy units of
    # correction on 1ubq.
    return _interior_angle_atan2(a - b, c - b)


@pt_interior_angle.defjvp
def _pt_interior_angle_jvp(primals, tangents):
    """JVP for :func:`pt_interior_angle` using upstream's analytic adjoint.

    Mirrors ``pt_interior_angle::V_dV`` in ``tmol/score/common/geom.hh``,
    which sets ``dV/dA = (BA/|BA|) x z_unit / |BA|``,
    ``dV/dC = -(BC/|BC|) x z_unit / |BC|``, ``dV/dB = -(dV/dA + dV/dC)``.

    All ``|...|`` denominators go through the double-where trick so
    the JVP is finite even on degenerate / padded inputs (zero-length
    BA or BC).
    """
    a, b, c = primals
    da, db, dc = tangents

    ba = a - b
    bc = c - b
    cr = jnp.cross(ba, bc)

    # Safe norms with the double-where pattern, identical to
    # _safe_norm. Need the values for both primal and tangent paths.
    a_sq = jnp.sum(ba * ba, axis=-1, keepdims=True)
    c_sq = jnp.sum(bc * bc, axis=-1, keepdims=True)
    cr_sq = jnp.sum(cr * cr, axis=-1, keepdims=True)

    safe_a = a_sq > _EPS_SQ
    safe_c = c_sq > _EPS_SQ
    safe_cr = cr_sq > _EPS_SQ

    a_norm = jnp.where(safe_a, jnp.sqrt(jnp.where(safe_a, a_sq, 1.0)), 1.0)
    c_norm = jnp.where(safe_c, jnp.sqrt(jnp.where(safe_c, c_sq, 1.0)), 1.0)
    cr_norm = jnp.where(safe_cr, jnp.sqrt(jnp.where(safe_cr, cr_sq, 1.0)), 1.0)

    # z_unit = CR / |CR|
    z_unit = cr / cr_norm  # masked below

    # Upstream formula:
    # dV_dA = (BA / |BA|).cross(z_unit) / |BA|
    # dV_dC = -(BC / |BC|).cross(z_unit) / |BC|
    # dV_dB = -(dV_dA + dV_dC)
    inv_a = 1.0 / a_norm  # safe
    inv_c = 1.0 / c_norm
    ba_unit = ba * inv_a
    bc_unit = bc * inv_c
    dV_dA = jnp.cross(ba_unit, z_unit) * inv_a
    dV_dC = -jnp.cross(bc_unit, z_unit) * inv_c
    dV_dB = -(dV_dA + dV_dC)

    # Mask out degenerate slots (no NaN cotangents propagate).
    finite_mask = (safe_a & safe_c & safe_cr).squeeze(-1)
    finite_mask_e = finite_mask[..., None]
    zero = jnp.zeros_like(dV_dA)
    dV_dA = jnp.where(finite_mask_e, dV_dA, zero)
    dV_dB = jnp.where(finite_mask_e, dV_dB, zero)
    dV_dC = jnp.where(finite_mask_e, dV_dC, zero)

    primal_out = pt_interior_angle(a, b, c)
    tangent_out = (
        jnp.sum(dV_dA * da, axis=-1)
        + jnp.sum(dV_dB * db, axis=-1)
        + jnp.sum(dV_dC * dc, axis=-1)
    )
    return primal_out, tangent_out


@jax.custom_jvp
def dihedral_angle(a: Array, b: Array, c: Array, d: Array) -> Array:
    """Signed atan2 dihedral, with zero derivatives for degenerate geometry.

    The analytical Blondel--Karplus JVP avoids differentiating normalizations.
    atan2 is better conditioned near planar angles than the native acos value;
    float32 gradients can consequently differ there despite equal energies.
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    u_bc = _safe_normalize(b2, axis=-1)

    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(jnp.cross(n1, n2) * u_bc, axis=-1)
    return _safe_atan2(y, x)


@dihedral_angle.defjvp
def _dihedral_angle_jvp(primals, tangents):
    """JVP using upstream's Blondel-Karplus analytical adjoint.

    Mirrors ``dihedral_angle<Real>::V_dV`` in ``geom.hh`` (Blondel,
    Karplus, J. Comput. Chem. 1996). Let::

        F = I - J
        G = J - K
        H = L - K
        A = F x G
        B = H x G

    then::

        dV_dI = -(|G| / A.A) * A
        dV_dJ =  (|G| / A.A) * A + (F.G / (A.A * |G|)) * A
                                  - (H.G / (B.B * |G|)) * B
        dV_dK = -(|G| / B.B) * B - (F.G / (A.A * |G|)) * A
                                  + (H.G / (B.B * |G|)) * B
        dV_dL =  (|G| / B.B) * B

    All denominators use the double-where pattern so the JVP is
    well-defined on degenerate inputs (zero-length G, colinear I-J-K
    or J-K-L).
    """
    I, J, K, L = primals
    dI, dJ, dK, dL = tangents

    F = I - J
    G = J - K
    H = L - K

    A = jnp.cross(F, G)
    B = jnp.cross(H, G)

    # Squared norms for the denominators; promote to keepdims=True so
    # the broadcast against the vectors A / B works cleanly.
    A2 = jnp.sum(A * A, axis=-1, keepdims=True)
    B2 = jnp.sum(B * B, axis=-1, keepdims=True)
    G2 = jnp.sum(G * G, axis=-1, keepdims=True)

    safe_A = A2 > _EPS_SQ
    safe_B = B2 > _EPS_SQ
    safe_G = G2 > _EPS_SQ

    # Double-where guards: replace zero radicands with 1 inside the
    # sqrt and zero divisor with 1 inside the divide so the cotangent
    # never sees inf / NaN on the masked branch.
    G_norm = jnp.where(safe_G, jnp.sqrt(jnp.where(safe_G, G2, 1.0)), 1.0)

    A2_safe = jnp.where(safe_A, A2, 1.0)
    B2_safe = jnp.where(safe_B, B2, 1.0)
    inv_A2 = 1.0 / A2_safe
    inv_B2 = 1.0 / B2_safe
    inv_G = 1.0 / G_norm

    FG = jnp.sum(F * G, axis=-1, keepdims=True)
    HG = jnp.sum(H * G, axis=-1, keepdims=True)

    # Upstream Blondel-Karplus closed form:
    coef_A_dI = -G_norm * inv_A2
    coef_A_extra = FG * inv_A2 * inv_G
    coef_B_dK = -G_norm * inv_B2
    coef_B_extra = HG * inv_B2 * inv_G

    dV_dI = coef_A_dI * A
    dV_dL = -coef_B_dK * B
    dV_dJ = -coef_A_dI * A + coef_A_extra * A - coef_B_extra * B
    dV_dK = coef_B_dK * B - coef_A_extra * A + coef_B_extra * B

    # Mask degenerate slots to zero so cotangents on padded entries
    # do not leak into the gradient.
    finite_mask_e = safe_A & safe_B & safe_G  # [..., 1]
    zero = jnp.zeros_like(dV_dI)
    dV_dI = jnp.where(finite_mask_e, dV_dI, zero)
    dV_dJ = jnp.where(finite_mask_e, dV_dJ, zero)
    dV_dK = jnp.where(finite_mask_e, dV_dK, zero)
    dV_dL = jnp.where(finite_mask_e, dV_dL, zero)

    primal_out = dihedral_angle(I, J, K, L)
    tangent_out = (
        jnp.sum(dV_dI * dI, axis=-1)
        + jnp.sum(dV_dJ * dJ, axis=-1)
        + jnp.sum(dV_dK * dK, axis=-1)
        + jnp.sum(dV_dL * dL, axis=-1)
    )
    return primal_out, tangent_out


__all__ = [
    "cos_interior_angle",
    "dihedral_angle",
    "distance",
    "normalize",
    "pt_interior_angle",
]
