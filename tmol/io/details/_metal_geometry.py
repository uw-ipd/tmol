"""Fit observed metal ligands to a coordination polyhedron.

Which geometry a metal has cannot be read off the number of ligands present.
Structure input drops waters, and a site left open by one is indistinguishable
from a site that was never there -- a magnesium with five waters and one
aspartate arrives as a single donor. So geometry is chosen by how well each
candidate polyhedron explains the directions of the donors that *are* present,
with the ion's preference order breaking ties the directions cannot.

The fit is over directions only: the polyhedron is free to rotate, so what is
compared is the shape the donors make around the metal, not where it points.
"""

from functools import lru_cache
from itertools import permutations
from typing import Optional, Sequence, Tuple

import attr
import numpy


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GeometryFit:
    """How well one candidate geometry explains a set of donor directions."""

    geometry: str
    # root-mean-square angle, degrees, between each donor and the vertex it was
    # matched to after the best rotation
    rms_angle: float
    # vertex index taken by each donor, in the order the donors were given
    vertex_for_donor: Tuple[int, ...]
    n_sites: int
    # carries the polyhedron's vertices onto the donors: vertex @ rotation.T
    rotation: Optional[numpy.ndarray] = attr.ib(default=None, eq=False)

    @property
    def n_open_sites(self) -> int:
        return self.n_sites - len(self.vertex_for_donor)


def unit(v: numpy.ndarray) -> numpy.ndarray:
    norm = numpy.linalg.norm(v, axis=-1, keepdims=True)
    return v / numpy.where(norm == 0, 1.0, norm)


def best_rotation(donors: numpy.ndarray, targets: numpy.ndarray) -> numpy.ndarray:
    """Rotation carrying ``donors`` onto ``targets``, both unit vectors (Kabsch)."""
    return _best_rotations(donors, targets[numpy.newaxis])[0]


def _best_rotations(donors: numpy.ndarray, targets: numpy.ndarray) -> numpy.ndarray:
    """Kabsch for a stack of target sets at once, ``targets`` shaped (n, k, 3)."""
    covariance = numpy.einsum("ki,nkj->nij", donors, targets)
    u, _, vt = numpy.linalg.svd(covariance)
    correction = numpy.zeros_like(u)
    correction[:, 0, 0] = correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = numpy.sign(numpy.linalg.det(u @ vt))
    return u @ correction @ vt


@lru_cache(maxsize=None)
def _assignments(n_sites: int, n_donors: int) -> numpy.ndarray:
    """Every injective donor-to-vertex map, as one (n, n_donors) index array."""
    return numpy.array(list(permutations(range(n_sites), n_donors)), dtype=int)


def fit_geometry(
    donor_directions: numpy.ndarray,
    vertices: numpy.ndarray,
) -> Optional[GeometryFit]:
    """Best assignment of donors to vertices, over all rotations of the polyhedron.

    Exhaustive over injective donor-to-vertex maps, which is affordable because
    no supported geometry has more than six vertices. Returns None when there
    are more donors than the geometry can hold.
    """
    donors = unit(numpy.asarray(donor_directions, dtype=numpy.float64))
    verts = unit(numpy.asarray(vertices, dtype=numpy.float64))
    n_donors, n_sites = len(donors), len(verts)
    if n_donors == 0 or n_donors > n_sites:
        return None

    candidates = _assignments(n_sites, n_donors)
    targets = verts[candidates]
    rotations = _best_rotations(donors, targets)
    rotated = numpy.einsum("ki,nij->nkj", donors, rotations)
    cosines = numpy.clip(numpy.sum(rotated * targets, axis=2), -1.0, 1.0)
    rms = numpy.sqrt(numpy.mean(numpy.degrees(numpy.arccos(cosines)) ** 2, axis=1))
    # Symmetric polyhedra fit equally well several ways, and batched SVD moves the
    # last few bits, so take the first of the tied assignments rather than whichever
    # one rounding favours.
    best = int(numpy.flatnonzero(rms <= rms.min() + 1e-9)[0])
    return GeometryFit(
        geometry="",
        rms_angle=float(rms[best]),
        vertex_for_donor=tuple(int(v) for v in candidates[best]),
        n_sites=n_sites,
        rotation=rotations[best],
    )


def choose_geometry(
    donor_directions: numpy.ndarray,
    allowed: Sequence[str],
    vertices_for: dict,
    tolerance_deg: float = 5.0,
) -> Tuple[Optional[GeometryFit], str]:
    """Pick the geometry that best explains the donors, and say how it was picked.

    Preference order in the ion's table is most-common first, and it decides
    every case the directions cannot: geometries fitting equally well within
    ``tolerance_deg``, and nested candidates where a larger polyhedron always
    fits at least as well as the smaller one it contains.

    Deliberately *not* biased toward the geometry positing fewest empty sites.
    That would be sound if a missing donor meant a missing site, but structure
    input drops waters, so sites are routinely empty for reasons that carry no
    information. Preferring the smaller polyhedron would make a magnesium with
    one aspartate and five dropped waters come out tetrahedral.
    """
    fits = []
    for rank, name in enumerate(allowed):
        verts = vertices_for[name]
        if not verts:  # untemplated: nothing to fit, and nothing to choose
            return None, "untemplated"
        fit = fit_geometry(donor_directions, numpy.asarray(verts))
        if fit is not None:
            fits.append(
                attr.evolve(fit, geometry=name),
            )
    if not fits:
        if len(donor_directions) == 0:
            return None, "no donors in range"
        return None, "no candidate geometry can hold this many donors"

    best_rms = min(f.rms_angle for f in fits)
    tied = [f for f in fits if f.rms_angle <= best_rms + tolerance_deg]
    if len(tied) == 1:
        return tied[0], "directions"
    return min(tied, key=lambda f: allowed.index(f.geometry)), "preference order"
