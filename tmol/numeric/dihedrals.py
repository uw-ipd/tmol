import numpy

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

CoordArray = NDArray(float)[:, 3]
Angles = NDArray(float)[:]


@validate_args
def coord_dihedrals(
        a: CoordArray, b: CoordArray, c: CoordArray, d: CoordArray
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

    ubc = bc / numpy.linalg.norm(bc, axis=-1).reshape((-1, 1))

    # v = projection of ba onto plane perpendicular to bc
    #     minus component that aligns with bc
    # w = projection of cd onto plane perpendicular to bc
    #     cd minus component that aligns with bc
    v = ba - numpy.sum(ba * ubc, axis=-1).reshape((-1, 1)) * ubc
    w = cd - numpy.sum(cd * ubc, axis=-1).reshape((-1, 1)) * ubc

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = numpy.sum(v * w, axis=-1)
    y = numpy.sum(numpy.cross(ubc, v) * w, axis=-1)

    return numpy.arctan2(y, x)
