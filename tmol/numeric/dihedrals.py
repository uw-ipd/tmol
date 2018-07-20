import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

CoordArray = Tensor(torch.double)[:, 3]
Angles = Tensor(float)[:]


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
    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python

    aprime = a.type(torch.double)
    bprime = b.type(torch.double)
    cprime = c.type(torch.double)
    dprime = d.type(torch.double)

    ba = aprime - bprime
    bc = cprime - bprime
    cd = dprime - cprime

    ubc = bc / torch.norm(bc, 2, dim=1, keepdim=True)

    # v = projection of ba onto plane perpendicular to bc
    #     minus component that aligns with bc
    # w = projection of cd onto plane perpendicular to bc
    #     cd minus component that aligns with bc
    v = ba - torch.sum(ba * ubc, dim=1).reshape((-1, 1)) * ubc
    w = cd - torch.sum(cd * ubc, dim=1).reshape((-1, 1)) * ubc

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.einsum("ij,ij->i", (v, w))
    y = torch.einsum("ij,ij->i", (torch.cross(ubc, v), w))

    return torch.atan2(y, x).type(torch.float)
