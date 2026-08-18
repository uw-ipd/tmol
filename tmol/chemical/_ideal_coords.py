import numpy
import numba


@numba.jit(nopython=True)
def eye4():
    """Create the identity homogeneous transform
    Only necessary because numpy.eye(4, dtype=numpy.float32)
    is strangely unsupported in numpy"""

    return numpy.eye(4, dtype=numpy.float32)

    # m = numpy.zeros((4, 4), dtype=numpy.float32)
    # m[0, 0] = 1
    # m[1, 1] = 1
    # m[2, 2] = 1
    # m[3, 3] = 1
    # return m


@numba.jit(nopython=True)
def normalize(v):
    return v / numpy.linalg.norm(v)


@numba.jit(nopython=True)
def frame_from_coords(p1, p2, p3):
    ht = eye4()
    z = normalize(p3 - p2)
    v21 = normalize(p1 - p2)
    y = normalize(v21 - numpy.dot(z, v21) * z)
    x = normalize(numpy.cross(y, z))

    ht[0:3, 0] = x
    ht[0:3, 1] = y
    ht[0:3, 2] = z
    ht[0:3, 3] = p3
    return ht


@numba.jit(nopython=True)
def rot_x(rot):
    ht = eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    # print("rot x", rot, crot, srot)
    ht[1, 1] = crot
    ht[2, 1] = srot
    ht[1, 2] = -srot
    ht[2, 2] = crot
    return ht


@numba.jit(nopython=True)
def rot_z(rot):
    ht = eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    # print("rot z", rot, crot, srot)
    ht[0, 0] = crot
    ht[1, 0] = srot
    ht[0, 1] = -srot
    ht[1, 1] = crot
    return ht


@numba.jit(nopython=True)
def trans_z(trans):
    ht = eye4()
    ht[2, 3] = trans
    return ht


@numba.jit(nopython=True)
def build_coords_from_icoors(icoors_ancestors, icoors_geom):
    # start with atom 1 at the origin
    # place atom 2 along the x axis
    # place atom 3 in the x-y plane
    # place all other atoms

    n_atoms = icoors_ancestors.shape[0]
    coords = numpy.zeros((n_atoms, 3), dtype=numpy.float32)
    coords[1, 0] = icoors_geom[1, 2]

    # coord 2 in the x-y plane. Atom 2 is bonded to either atom 1 or atom 0
    # (its only two possible predecessors); its parent is whichever it is
    # bonded to and the geometry (d, theta) is measured against that parent.
    #
    # parent == atom 1 (chain 0->1->2, the canonical case): walk d from atom 1,
    #   angle taken against the atom1->atom0 direction.
    # parent == atom 0 (atom 2 is a sibling of atom 1 off the root): walk d from
    #   the origin, angle taken against the atom0->atom1 direction (rotate by
    #   pi - theta so atom 2 still lands in the +y half plane).
    if icoors_ancestors[2, 0] == 1:
        ht_base = eye4()
        ht_base[:3, 3] = coords[1, :]
        rot_2 = rot_z(icoors_geom[2, 1])
    else:
        ht_base = eye4()
        ht_base[:3, 3] = coords[0, :]
        rot_2 = rot_z(numpy.pi - icoors_geom[2, 1])
    trans_2 = eye4()
    trans_2[0, 3] = icoors_geom[2, 2]
    ht_2 = ht_base @ rot_2 @ trans_2

    coords[2, :] = ht_2[:3, 3]

    for i in range(3, icoors_ancestors.shape[0]):
        # print("ancestors", i)
        ht_i = frame_from_coords(
            coords[icoors_ancestors[i, 2], :],
            coords[icoors_ancestors[i, 1], :],
            coords[icoors_ancestors[i, 0], :],
        )

        ht_rot_z = rot_z(icoors_geom[i, 0])
        # negative rotation about the x axis will point z
        # at the next atom
        ht_rot_x = rot_x(-icoors_geom[i, 1])
        ht_trans_z = trans_z(icoors_geom[i, 2])

        ht_i = ht_i @ ht_rot_z @ ht_rot_x @ ht_trans_z
        coords[i, :3] = ht_i[:3, 3]
    return coords


def build_ideal_coords(restype: "RefinedResidueType"):  # noqa F821
    # lets build a kinforest using not the prioritized bonds,
    # but the icoors; let's not even use the scan algorithm.
    # let's just build the coordinates directly from the
    # tree provided in the icoors.

    return build_coords_from_icoors(restype.icoors_ancestors, restype.icoors_geom)
