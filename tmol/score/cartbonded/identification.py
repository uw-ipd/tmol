import attr

from tmol.types.functional import convert_args
from tmol.types.array import NDArray
from tmol.database.scoring import CartBondedDatabase

from ..bonded_atom import IndexedBonds

import numpy

from numba import jit


# traverse bond graph, generate angles
@jit(nopython=True)
def find_angles(bonds, bond_spans):
    nbonds = bonds.shape[0]
    max_angles = int(nbonds * 3 / 2)  # assumes each node has at most 4 connections
    angles = numpy.zeros((max_angles, 3), dtype=numpy.int32)
    nangles = 0

    # use atm1:atm3 ordering to ensure no duplication
    for i in range(nbonds):
        atm1, atm2 = bonds[i]
        for j in range(bond_spans[atm2, 0], bond_spans[atm2, 1]):
            atm3 = bonds[j, 1]
            if atm3 > atm1:
                angles[nangles, :] = [atm1, atm2, atm3]
                nangles += 1

    return angles[:nangles]


# traverse bond graph, generate torsions
@jit(nopython=True)
def find_torsions(bonds, bond_spans):
    nbonds = bonds.shape[0]
    max_torsions = int(nbonds * 9 / 2)  # assumes each node has at most 4 connections
    torsions = numpy.zeros((max_torsions, 4), dtype=numpy.int32)
    ntorsions = 0

    # use atm0:atm3 ordering to ensure no duplication
    # note: cycles like A-B-C-A are valid!
    for i in range(nbonds):
        atm1, atm2 = bonds[i]

        for j in range(bond_spans[atm2, 0], bond_spans[atm2, 1]):
            atm3 = bonds[j, 1]
            if atm3 == atm1:
                continue

            for k in range(bond_spans[atm1, 0], bond_spans[atm1, 1]):
                atm0 = bonds[k, 1]
                if atm0 == atm2:
                    continue

                if (atm0 < atm3) or (atm0 == atm3 and atm1 < atm2):
                    torsions[ntorsions, :] = [atm0, atm1, atm2, atm3]
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
def find_impropers(bonds, bond_spans):
    nbonds = bonds.shape[0]
    max_impropers = nbonds * 6  # assumes each node has at most 4 connections
    improper = numpy.zeros((max_impropers, 4), dtype=numpy.int32)
    nimproper = 0

    # note: ABCD and BACD are considered separate terms
    for i in range(nbonds):
        atm1, atm2 = bonds[i]

        for j in range(bond_spans[atm2, 0], bond_spans[atm2, 1]):
            atm3 = bonds[j, 1]
            if atm3 == atm1:
                continue
            for k in range(bond_spans[atm2, 0], bond_spans[atm2, 1]):
                atm4 = bonds[k, 1]
                if atm4 == atm1 or atm4 == atm3:
                    continue

                improper[nimproper, :] = [atm4, atm3, atm2, atm1]
                nimproper += 1

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
    def setup(
        cls, cartbonded_database: CartBondedDatabase, indexed_bonds: IndexedBonds
    ):
        bonds = indexed_bonds.bonds.cpu().numpy()
        spans = indexed_bonds.bond_spans.cpu().numpy()
        bond_selector = bonds[:, 0] < bonds[:, 1]

        lengths = bonds[bond_selector].copy()
        angles = find_angles(bonds, spans)
        torsions = find_torsions(bonds, spans)
        impropers = find_impropers(bonds, spans)

        return cls(
            lengths=lengths, angles=angles, torsions=torsions, impropers=impropers
        )
