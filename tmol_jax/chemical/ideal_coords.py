"""Ideal coordinate construction from ``Icoor`` records.

Ported from ``tmol.chemical.ideal_coords``. The original implementation
used ``numba.jit``; here we use pure numpy so we don't need numba as a
runtime dependency. Ideal-coordinate construction is a one-time
pre-processing step (per residue type), so performance is not critical.

The algorithm walks the icoor tree (atoms in topological order from root)
and places each atom using the (phi, theta, d) parameters of its icoor
record relative to its parent, grand-parent, and great-grand-parent
coordinate frame:

  * Atom 0 is placed at the origin.
  * Atom 1 is placed along the x-axis at distance ``d[1]`` from atom 0.
  * Atom 2 is placed in the x-y plane, rotated by ``theta[2]`` about the
    z-axis and translated ``d[2]`` along the rotated x-axis.
  * Atom i >= 3 is placed at a homogeneous transform built from the frame
    defined by (great_grand_parent, grand_parent, parent), followed by a
    rotation ``rot_z(phi)``, a negative rotation ``rot_x(-theta)``, and a
    translation of ``d`` along z.
"""

from __future__ import annotations

import numpy as np


def eye4() -> np.ndarray:
    """4x4 identity homogeneous transform, ``float32``."""
    return np.eye(4, dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit length."""
    return v / np.linalg.norm(v)


def frame_from_coords(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous frame located at ``p3``.

    The frame's z-axis points from ``p2`` to ``p3``, its y-axis is the
    component of (p1 - p2) orthogonal to z, and its x-axis is ``y x z``.
    """
    ht = eye4()
    z = normalize(p3 - p2)
    v21 = normalize(p1 - p2)
    y = normalize(v21 - np.dot(z, v21) * z)
    x = normalize(np.cross(y, z))

    ht[0:3, 0] = x
    ht[0:3, 1] = y
    ht[0:3, 2] = z
    ht[0:3, 3] = p3
    return ht


def rot_x(rot: float) -> np.ndarray:
    """Rotation about the x-axis by ``rot`` radians."""
    ht = eye4()
    crot = np.cos(rot)
    srot = np.sin(rot)
    ht[1, 1] = crot
    ht[2, 1] = srot
    ht[1, 2] = -srot
    ht[2, 2] = crot
    return ht


def rot_z(rot: float) -> np.ndarray:
    """Rotation about the z-axis by ``rot`` radians."""
    ht = eye4()
    crot = np.cos(rot)
    srot = np.sin(rot)
    ht[0, 0] = crot
    ht[1, 0] = srot
    ht[0, 1] = -srot
    ht[1, 1] = crot
    return ht


def trans_z(trans: float) -> np.ndarray:
    """Translation along the z-axis by ``trans``."""
    ht = eye4()
    ht[2, 3] = trans
    return ht


def build_coords_from_icoors(
    icoors_ancestors: np.ndarray,
    icoors_geom: np.ndarray,
) -> np.ndarray:
    """Build 3-D coordinates from an icoor tree.

    Parameters
    ----------
    icoors_ancestors: ``int32[n_atoms, 3]``
        For each icoor (index ``i``), the ancestor indices
        ``(parent, grand_parent, great_grand_parent)``. Indices refer to
        rows of the same array.
    icoors_geom: ``float64[n_atoms, 3]``
        For each icoor, ``(phi, theta, d)`` where ``phi`` is the dihedral
        from the ``great_grand_parent -> grand_parent -> parent -> atom``,
        ``theta`` is the ``grand_parent -> parent -> atom`` bond angle,
        and ``d`` is the distance from parent to atom.

    Returns
    -------
    coords: ``float32[n_atoms, 3]``
    """
    n_atoms = icoors_ancestors.shape[0]
    coords = np.zeros((n_atoms, 3), dtype=np.float32)

    if n_atoms <= 1:
        return coords

    # Atom 1 on the +x axis at distance d[1].
    coords[1, 0] = icoors_geom[1, 2]

    if n_atoms <= 2:
        return coords

    # Atom 2 in the x-y plane: put a frame at p1 (aligned with world axes),
    # rotate by theta[2] about z so the frame's x points where p2 goes,
    # then walk d[2] along that x.
    ht_1 = eye4()
    parent = int(icoors_ancestors[2, 0])
    ht_1[:3, 3] = coords[parent, :]
    theta = icoors_geom[2, 1] if parent == 1 else np.pi - icoors_geom[2, 1]
    rot_2 = rot_z(theta)
    trans_2 = eye4()
    trans_2[0, 3] = icoors_geom[2, 2]
    ht_2 = ht_1 @ rot_2 @ trans_2
    coords[2, :] = ht_2[:3, 3]

    # Atoms 3..n: build a local frame from (ggp, gp, p) and apply
    # rot_z(phi) @ rot_x(-theta) @ trans_z(d) to get the child position.
    for i in range(3, n_atoms):
        ht_i = frame_from_coords(
            coords[icoors_ancestors[i, 2], :],
            coords[icoors_ancestors[i, 1], :],
            coords[icoors_ancestors[i, 0], :],
        )
        ht_rot_z = rot_z(icoors_geom[i, 0])
        ht_rot_x = rot_x(-icoors_geom[i, 1])
        ht_trans_z = trans_z(icoors_geom[i, 2])
        ht_i = ht_i @ ht_rot_z @ ht_rot_x @ ht_trans_z
        coords[i, :3] = ht_i[:3, 3]

    return coords


def build_ideal_coords(restype) -> np.ndarray:
    """Build ideal coordinates from a ``RefinedResidueType``.

    Delegates to :func:`build_coords_from_icoors` using the residue's
    pre-computed ancestor index / geometry arrays.
    """
    return build_coords_from_icoors(restype.icoors_ancestors, restype.icoors_geom)


__all__ = [
    "eye4",
    "normalize",
    "frame_from_coords",
    "rot_x",
    "rot_z",
    "trans_z",
    "build_coords_from_icoors",
    "build_ideal_coords",
]
