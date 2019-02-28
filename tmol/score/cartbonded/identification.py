import attr

from tmol.types.functional import convert_args
from tmol.types.array import NDArray
from tmol.database.scoring import CartBondedDatabase

import numpy

from numba import jit


## fd TODO: replace this with fast IndexedAtom versions

# traverse bond graph, generate angles
@jit(nopython=True)
def find_angles(bonds):
    nbonds = bonds.shape[0]
    max_angles = nbonds * 3  # assume each node has at most 4 connections
    angles = numpy.zeros((max_angles, 3), dtype=numpy.int32)
    nangles = 0

    # use ordering on bonds to ensure no duplication
    for i in range(nbonds):
        for j in range(i + 1, nbonds):
            if bonds[i, 0] == bonds[j, 0] and bonds[i, 1] != bonds[j, 1]:
                angles[nangles, :] = [bonds[i, 1], bonds[i, 0], bonds[j, 1]]
                nangles += 1
            elif bonds[i, 0] == bonds[j, 1] and bonds[i, 1] != bonds[j, 0]:
                angles[nangles, :] = [bonds[i, 1], bonds[i, 0], bonds[j, 0]]
                nangles += 1
            elif bonds[i, 1] == bonds[j, 0] and bonds[i, 0] != bonds[j, 1]:
                angles[nangles, :] = [bonds[i, 0], bonds[i, 1], bonds[j, 1]]
                nangles += 1
            elif bonds[i, 1] == bonds[j, 1] and bonds[i, 0] != bonds[j, 0]:
                angles[nangles, :] = [bonds[i, 0], bonds[i, 1], bonds[j, 0]]
                nangles += 1

    return angles[:nangles]


# traverse bond graph, generate torsions
@jit(nopython=True)
def find_torsions(angles, bonds):
    nbonds = bonds.shape[0]
    nangles = angles.shape[0]
    max_torsions = nangles * 3  # assume each node has at most 4 connections
    torsions = numpy.zeros((max_torsions, 4), dtype=numpy.int32)
    ntorsions = 0

    # note 1: we order atm1 and atm4 to prevent duplication
    # note 2: we do not check for cycles A-B-C-A because we want those
    #   to be generated!
    for i in range(nangles):
        for j in range(nbonds):
            if (
                angles[i, 0] == bonds[j, 0]
                and angles[i, 1] != bonds[j, 1]
                and bonds[j, 1] < angles[i, 2]
            ):
                torsions[ntorsions, :] = [
                    bonds[j, 1],
                    angles[i, 0],
                    angles[i, 1],
                    angles[i, 2],
                ]
                ntorsions += 1
            elif (
                angles[i, 0] == bonds[j, 1]
                and angles[i, 1] != bonds[j, 0]
                and bonds[j, 0] < angles[i, 2]
            ):
                torsions[ntorsions, :] = [
                    bonds[j, 0],
                    angles[i, 0],
                    angles[i, 1],
                    angles[i, 2],
                ]
                ntorsions += 1
            elif (
                angles[i, 2] == bonds[j, 0]
                and angles[i, 1] != bonds[j, 1]
                and bonds[j, 1] < angles[i, 0]
            ):
                torsions[ntorsions, :] = [
                    bonds[j, 1],
                    angles[i, 2],
                    angles[i, 1],
                    angles[i, 0],
                ]
                ntorsions += 1
            elif (
                angles[i, 2] == bonds[j, 1]
                and angles[i, 1] != bonds[j, 0]
                and bonds[j, 0] < angles[i, 0]
            ):
                torsions[ntorsions, :] = [
                    bonds[j, 0],
                    angles[i, 2],
                    angles[i, 1],
                    angles[i, 0],
                ]
                ntorsions += 1

    return torsions[:ntorsions]


# traverse bond graph, generate improper torsions
#   ABCD combinations with bond pattern:
#         A
#          \
#           C-D
#          /
#         B
@jit(nopython=True)
def find_impropers(angles, bonds):
    nbonds = bonds.shape[0]
    nangles = angles.shape[0]
    max_impropers = 4 * nangles
    improper = numpy.zeros((max_impropers, 4), dtype=numpy.int32)
    nimproper = 0

    # impropers are a bit strange, we need to consider both ABCD and BACD as
    #  separate terms, as both might be constrained separately
    for i in range(nangles):
        for j in range(nbonds):
            if (
                angles[i, 1] == bonds[j, 0]
                and angles[i, 0] != bonds[j, 1]
                and angles[i, 2] != bonds[j, 1]
            ):
                improper[nimproper, :] = [
                    angles[i, 0],
                    angles[i, 2],
                    angles[i, 1],
                    bonds[j, 1],
                ]
                improper[nimproper + 1, :] = [
                    angles[i, 2],
                    angles[i, 0],
                    angles[i, 1],
                    bonds[j, 1],
                ]
                nimproper += 2
            elif (
                angles[i, 1] == bonds[j, 1]
                and angles[i, 0] != bonds[j, 0]
                and angles[i, 2] != bonds[j, 0]
            ):
                improper[nimproper, :] = [
                    angles[i, 0],
                    angles[i, 2],
                    angles[i, 1],
                    bonds[j, 0],
                ]
                improper[nimproper + 1, :] = [
                    angles[i, 2],
                    angles[i, 0],
                    angles[i, 1],
                    bonds[j, 0],
                ]
                nimproper += 2

    return improper[:nimproper]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class CartBondedIdentification:
    """Expands a bondgraph to export all bondangles and bondlengths
    """

    lengths: NDArray(int)[:, 2]
    angles: NDArray(int)[:, 3]
    torsions: NDArray(int)[:, 4]
    impropers: NDArray(int)[:, 4]

    @classmethod
    @convert_args
    def setup(cls, cartbonded_database: CartBondedDatabase, bonds: NDArray(int)[:, 2]):
        lengths = bonds[bonds[:, 0] < bonds[:, 1]]  # triu
        angles = find_angles(lengths)
        torsions = find_torsions(angles, lengths)
        impropers = find_impropers(angles, lengths)

        return cls(
            lengths=lengths, angles=angles, torsions=torsions, impropers=impropers
        )
