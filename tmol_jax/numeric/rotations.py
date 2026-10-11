"""3-D rotation utilities in pure JAX.

All routines accept and return :class:`jax.Array`-compatible values and
are fully differentiable through :func:`jax.grad` / :func:`jax.jvp`.
Shapes are documented per-function; the implementations broadcast over
arbitrary leading batch dimensions so they compose with :func:`jax.vmap`
without any further wrapping.

Conventions
-----------
* Matrices operate on *column* vectors (``v' = R @ v``) — matches the
  ``Eigen::Matrix3d`` convention used throughout upstream tmol.
* Euler ZYX: ``R = Rz(alpha) @ Ry(beta) @ Rx(gamma)``; applied to a
  point as ``Rz(Ry(Rx v))``. This is the same convention as
  ``tmol_jax.kinematics.operations._euler_zyx_delta``.
* Quaternions are stored as ``[w, x, y, z]`` (scalar-first). Pure
  rotations require unit quaternions; helpers below normalise on
  demand.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Axis-angle
# ---------------------------------------------------------------------------


def rotation_matrix_from_axis_angle(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotation matrix from an axis-angle pair via Rodrigues' formula.

    Parameters
    ----------
    axis : ``[..., 3]`` array
        Rotation axis. Need not be unit length; it is normalised
        internally with a numerically stable eps-guard so a zero axis
        gives the identity with a well-defined gradient.
    angle : ``[...]`` scalar
        Rotation angle in radians. Broadcast-compatible with ``axis``.

    Returns
    -------
    R : ``[..., 3, 3]`` array
        Rotation matrix satisfying ``v' = R @ v`` with ``v`` a column
        vector. Equivalent to :math:`I + \\sin\\theta\\,K + (1 - \\cos\\theta) K^2`
        where ``K`` is the cross-product matrix of the unit axis.
    """

    axis = jnp.asarray(axis)
    angle = jnp.asarray(angle)

    # Stable unit axis. jnp.linalg.norm has a 0-gradient singularity at the
    # origin; the eps guard keeps grads finite. Zero-angle rotations are
    # still handled correctly because the sin/cos factors vanish.
    eps = jnp.asarray(1e-12, dtype=axis.dtype)
    norm = jnp.sqrt(jnp.sum(axis * axis, axis=-1, keepdims=True) + eps)
    u = axis / norm  # [..., 3]

    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]
    zero = jnp.zeros_like(ux)

    # Cross-product matrix K such that K @ v = u x v
    K = jnp.stack(
        [
            jnp.stack([zero, -uz, uy], axis=-1),
            jnp.stack([uz, zero, -ux], axis=-1),
            jnp.stack([-uy, ux, zero], axis=-1),
        ],
        axis=-2,
    )

    I3 = jnp.broadcast_to(jnp.eye(3, dtype=axis.dtype), K.shape)
    sin_a = jnp.sin(angle)[..., None, None]
    cos_a = jnp.cos(angle)[..., None, None]

    # Rodrigues: R = I + sin(a) K + (1 - cos(a)) K^2
    return I3 + sin_a * K + (1.0 - cos_a) * jnp.matmul(K, K)


# ---------------------------------------------------------------------------
# Euler ZYX
# ---------------------------------------------------------------------------


def _rot_x(angle: jax.Array) -> jax.Array:
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([o, z, z], axis=-1),
            jnp.stack([z, c, -s], axis=-1),
            jnp.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


def _rot_y(angle: jax.Array) -> jax.Array:
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, z, s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([-s, z, c], axis=-1),
        ],
        axis=-2,
    )


def _rot_z(angle: jax.Array) -> jax.Array:
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, -s, z], axis=-1),
            jnp.stack([s, c, z], axis=-1),
            jnp.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def euler_zyx_to_matrix(
    alpha: jax.Array, beta: jax.Array, gamma: jax.Array
) -> jax.Array:
    """Euler ZYX -> rotation matrix.

    Returns ``Rz(alpha) @ Ry(beta) @ Rx(gamma)``.

    Parameters
    ----------
    alpha, beta, gamma : broadcast-compatible scalar arrays
        Rotation angles (in radians) about the Z, Y, and X axes
        respectively, applied right-to-left (Rx first).

    Returns
    -------
    ``[..., 3, 3]`` array
    """

    Rz = _rot_z(jnp.asarray(alpha))
    Ry = _rot_y(jnp.asarray(beta))
    Rx = _rot_x(jnp.asarray(gamma))
    return jnp.matmul(Rz, jnp.matmul(Ry, Rx))


def matrix_to_euler_zyx(R: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Extract ZYX Euler angles from a rotation matrix.

    Returns ``(alpha, beta, gamma)`` such that
    ``R == Rz(alpha) @ Ry(beta) @ Rx(gamma)`` away from the
    ``cos(beta) == 0`` gimbal-lock singularity. At the singularity the
    function still returns finite angles and recovers ``beta = +/- pi/2``
    with ``gamma = 0`` (all yaw absorbed into ``alpha``), mirroring the
    convention in :mod:`tmol_jax.kinematics.operations`.
    """

    R = jnp.asarray(R)
    # beta = arcsin(-R[2, 0]); clip for safety against FP drift.
    sin_beta = -R[..., 2, 0]
    sin_beta = jnp.clip(sin_beta, -1.0, 1.0)
    beta = jnp.arcsin(sin_beta)
    cos_beta = jnp.cos(beta)

    eps = 1e-6
    gimbal = jnp.abs(cos_beta) < eps

    # Non-degenerate branch.
    alpha_reg = jnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    gamma_reg = jnp.arctan2(R[..., 2, 1], R[..., 2, 2])

    # Gimbal branch: put all yaw into alpha, gamma = 0.
    alpha_gim = jnp.arctan2(-R[..., 0, 1], R[..., 1, 1])
    gamma_gim = jnp.zeros_like(gamma_reg)

    alpha = jnp.where(gimbal, alpha_gim, alpha_reg)
    gamma = jnp.where(gimbal, gamma_gim, gamma_reg)
    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def quaternion_normalize(q: jax.Array) -> jax.Array:
    """Return ``q / ||q||`` with a small eps for stability."""
    q = jnp.asarray(q)
    eps = jnp.asarray(1e-12, dtype=q.dtype)
    return q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + eps)


def quaternion_conjugate(q: jax.Array) -> jax.Array:
    """Quaternion conjugate ``(w, -x, -y, -z)``. For unit ``q`` this is the inverse."""
    q = jnp.asarray(q)
    return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quaternion_multiply(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product ``a * b`` (scalar-first convention).

    ``rotation(a * b) = rotation(a) @ rotation(b)``.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def matrix_from_quaternion(q: jax.Array) -> jax.Array:
    """Unit quaternion (``w, x, y, z``) -> rotation matrix.

    Input does not need to be pre-normalised; the formula is applied
    after normalisation so the gradient is well-defined.
    """
    q = quaternion_normalize(jnp.asarray(q))
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = jnp.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
    row1 = jnp.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
    row2 = jnp.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def quaternion_from_matrix(R: jax.Array) -> jax.Array:
    """Rotation matrix -> unit quaternion (``w, x, y, z``).

    Uses Shepperd's branching method [1]_ to pick the quaternion
    component with the largest magnitude first, then derives the others
    by direct algebra. This avoids the instability of always taking
    ``w = 0.5 * sqrt(1 + trace)`` (which degrades when ``trace`` is
    close to ``-1``) while remaining branch-free (all four branches are
    evaluated and selected with :func:`jnp.where`) so :func:`jax.jit`
    is happy.

    .. [1] Shepperd, S.W. (1978) *Quaternion from Rotation Matrix*,
           J. Guidance and Control, 1(3), 223-224.
    """

    R = jnp.asarray(R)
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    # Candidate magnitudes (4 * q_i^2) for each component.
    t_w = 1.0 + trace
    t_x = 1.0 + m00 - m11 - m22
    t_y = 1.0 - m00 + m11 - m22
    t_z = 1.0 - m00 - m11 + m22

    # Pick the largest to avoid catastrophic cancellation.
    ts = jnp.stack([t_w, t_x, t_y, t_z], axis=-1)  # [..., 4]
    which = jnp.argmax(ts, axis=-1)  # [...]

    # Guard sqrt argument against tiny negatives from FP drift.
    sqrt_tw = jnp.sqrt(jnp.maximum(t_w, 0.0))
    sqrt_tx = jnp.sqrt(jnp.maximum(t_x, 0.0))
    sqrt_ty = jnp.sqrt(jnp.maximum(t_y, 0.0))
    sqrt_tz = jnp.sqrt(jnp.maximum(t_z, 0.0))

    # Branch 0: w-dominant.
    s_w = 2.0 * sqrt_tw  # = 4w
    q0 = jnp.stack(
        [
            0.5 * sqrt_tw,
            (R[..., 2, 1] - R[..., 1, 2]) / s_w,
            (R[..., 0, 2] - R[..., 2, 0]) / s_w,
            (R[..., 1, 0] - R[..., 0, 1]) / s_w,
        ],
        axis=-1,
    )

    # Branch 1: x-dominant.
    s_x = 2.0 * sqrt_tx  # = 4x
    q1 = jnp.stack(
        [
            (R[..., 2, 1] - R[..., 1, 2]) / s_x,
            0.5 * sqrt_tx,
            (R[..., 0, 1] + R[..., 1, 0]) / s_x,
            (R[..., 0, 2] + R[..., 2, 0]) / s_x,
        ],
        axis=-1,
    )

    # Branch 2: y-dominant.
    s_y = 2.0 * sqrt_ty  # = 4y
    q2 = jnp.stack(
        [
            (R[..., 0, 2] - R[..., 2, 0]) / s_y,
            (R[..., 0, 1] + R[..., 1, 0]) / s_y,
            0.5 * sqrt_ty,
            (R[..., 1, 2] + R[..., 2, 1]) / s_y,
        ],
        axis=-1,
    )

    # Branch 3: z-dominant.
    s_z = 2.0 * sqrt_tz  # = 4z
    q3 = jnp.stack(
        [
            (R[..., 1, 0] - R[..., 0, 1]) / s_z,
            (R[..., 0, 2] + R[..., 2, 0]) / s_z,
            (R[..., 1, 2] + R[..., 2, 1]) / s_z,
            0.5 * sqrt_tz,
        ],
        axis=-1,
    )

    # Stack candidates then gather along the selected branch.
    candidates = jnp.stack([q0, q1, q2, q3], axis=-2)  # [..., 4, 4]
    idx = which[..., None, None]  # [..., 1, 1]
    q = jnp.take_along_axis(
        candidates, jnp.broadcast_to(idx, (*which.shape, 1, 4)), axis=-2
    )[..., 0, :]

    # Canonicalise sign so that w >= 0 (rotation double-cover).
    sign = jnp.where(q[..., 0] < 0, -1.0, 1.0)[..., None]
    return q * sign


__all__ = [
    "rotation_matrix_from_axis_angle",
    "euler_zyx_to_matrix",
    "matrix_to_euler_zyx",
    "quaternion_from_matrix",
    "matrix_from_quaternion",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_normalize",
]
