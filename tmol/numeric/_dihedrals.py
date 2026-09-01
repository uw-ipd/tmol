import numpy
import torch

from tmol.types import (
    NDArray,
    validate_args,
    Tensor,
)

Coord64Array = Tensor[torch.double][:, 3]
Angles = Tensor[float][:]


def _numpy_coord_dihedrals(
    a: NDArray[float][..., 3],
    b: NDArray[float][..., 3],
    c: NDArray[float][..., 3],
    d: NDArray[float][..., 3],
) -> NDArray[numpy.float32][...]:
    """Return vectorized coordinate dihedrals for NumPy topology setup."""
    ba = a - b
    bc = c - b
    cd = d - c
    unit_bc = bc / numpy.linalg.norm(bc, axis=-1, keepdims=True)
    v = ba - numpy.sum(ba * unit_bc, axis=-1, keepdims=True) * unit_bc
    w = cd - numpy.sum(cd * unit_bc, axis=-1, keepdims=True) * unit_bc
    x = numpy.sum(v * w, axis=-1)
    y = numpy.sum(numpy.cross(unit_bc, v) * w, axis=-1)
    return numpy.asarray(numpy.arctan2(y, x), dtype=numpy.float32)


@validate_args
def coord_dihedrals(
    a: Coord64Array, b: Coord64Array, c: Coord64Array, d: Coord64Array
) -> Angles:
    """Dihedral angle in [-pi, pi] over the planes defined by {a, b, c} & {b, c, d}.

    Calculate dihedral angle from four coordinate locations, using the
    "standard" torsion angle definition of two planes defined by the points
    {a, b, c} and {b, c, d}. For a four-atom bond definition, this corrosponds
    to rotation about the b-c bond.
    """

    # Implementation derived from the "Praxeolitic" method, described at
    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python # noqa

    ba = a - b
    bc = c - b
    cd = d - c

    ubc = bc / torch.norm(bc, 2, dim=1, keepdim=True)

    # v = projection of ba onto plane perpendicular to bc
    #     minus component that aligns with bc
    # w = projection of cd onto plane perpendicular to bc
    #     cd minus component that aligns with bc
    v = ba - torch.sum(ba * ubc, dim=1).reshape((-1, 1)) * ubc
    w = cd - torch.sum(cd * ubc, dim=1).reshape((-1, 1)) * ubc

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.linalg.cross(ubc, v) * w, dim=1)

    return torch.atan2(y, x).type(torch.float)
