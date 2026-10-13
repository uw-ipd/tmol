"""JAX port of :func:`tmol.numeric.dihedrals.coord_dihedrals`.

Given four batched coordinate arrays, returns the signed dihedral angle
in radians over the planes ``(a, b, c)`` and ``(b, c, d)`` using the
"Praxeolitic" projection formulation (numerically stable for small or
large angles and a drop-in replacement for the torch routine)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def coord_dihedrals(
    a: jax.Array, b: jax.Array, c: jax.Array, d: jax.Array
) -> jax.Array:
    """Signed dihedral angle in ``(-pi, pi]`` for each row of inputs.

    Parameters
    ----------
    a, b, c, d : ``[N, 3]`` arrays (or broadcast-compatible)
        Atom positions defining the dihedral a-b-c-d.

    Returns
    -------
    ``[N]`` array of dihedrals in radians. Non-finite coordinates
    propagate to non-finite outputs exactly as the upstream routine
    does.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b)
    c = jnp.asarray(c)
    d = jnp.asarray(d)

    ba = a - b
    bc = c - b
    cd = d - c

    # Unit vector along the central bond. Add a tiny eps for stability;
    # degenerate (b == c) inputs are already non-physical.
    eps = jnp.asarray(1e-20, dtype=bc.dtype)
    ubc = bc / jnp.sqrt(jnp.sum(bc * bc, axis=-1, keepdims=True) + eps)

    # Project ba and cd into the plane perpendicular to bc.
    v = ba - jnp.sum(ba * ubc, axis=-1, keepdims=True) * ubc
    w = cd - jnp.sum(cd * ubc, axis=-1, keepdims=True) * ubc

    # Torsion angle = angle between v and w around ubc.
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(ubc, v) * w, axis=-1)
    return jnp.arctan2(y, x)


__all__ = ["coord_dihedrals"]
